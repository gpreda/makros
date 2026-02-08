"""Local landmark-based face/body morphing for progress video generation.

Uses MediaPipe for landmark detection and Delaunay triangulation for
geometric warping between consecutive progress photos.
"""

import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from pathlib import Path
from typing import Optional

# Output video settings
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280
FPS = 30
FRAMES_PER_TRANSITION = 45  # 1.5s morph between each photo pair
HOLD_FRAMES = 15            # 0.5s pause on each photo stage

# Model paths (relative to this file)
_MODELS_DIR = Path(__file__).parent / "models"
_FACE_MODEL = str(_MODELS_DIR / "face_landmarker.task")
_POSE_MODEL = str(_MODELS_DIR / "pose_landmarker.task")

# MediaPipe landmark indices for upper body pose points
_UPPER_BODY_POSE_INDICES = [
    0,                  # nose
    11, 12,             # shoulders
    13, 14,             # elbows
    15, 16,             # wrists
    23, 24,             # hips
]


def detect_landmarks(image_bgr: np.ndarray) -> Optional[dict]:
    """Detect face mesh and pose landmarks on an image.

    Returns dict with 'face' (Nx2) and 'pose' (Nx2) arrays in pixel coords,
    or None if detection fails entirely.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = {}

    # Face landmarks
    face_opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_FACE_MODEL),
        num_faces=1,
    )
    face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_opts)
    face_result = face_landmarker.detect(mp_image)
    if face_result.face_landmarks:
        lm = face_result.face_landmarks[0]
        result['face'] = np.array(
            [(p.x * w, p.y * h) for p in lm], dtype=np.float64
        )
    face_landmarker.close()

    # Pose landmarks
    pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_POSE_MODEL),
        num_poses=1,
    )
    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts)
    pose_result = pose_landmarker.detect(mp_image)
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        result['pose'] = np.array(
            [(p.x * w, p.y * h) for p in lm], dtype=np.float64
        )
    pose_landmarker.close()

    return result if result else None


def compute_upper_body_crop(landmarks: dict, img_w: int, img_h: int) -> tuple:
    """Compute a 9:16 crop rectangle for upper body based on landmarks.

    Returns (x, y, w, h) in pixel coordinates.
    """
    if 'pose' in landmarks:
        pose = landmarks['pose']
        # Key points: nose(0), shoulders(11,12), hips(23,24)
        nose = pose[0]
        l_shoulder, r_shoulder = pose[11], pose[12]
        l_hip, r_hip = pose[23], pose[24]

        # Vertical: from above head to hips
        mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        head_margin = (mid_shoulder_y - nose[1]) * 1.5
        top = nose[1] - head_margin
        bottom = max(l_hip[1], r_hip[1]) + 30  # small padding below hips

        # Horizontal: from outer shoulders with padding
        left = min(l_shoulder[0], r_shoulder[0], pose[13][0], pose[14][0])
        right = max(l_shoulder[0], r_shoulder[0], pose[13][0], pose[14][0])
        shoulder_width = right - left
        padding = shoulder_width * 0.4
        left -= padding
        right += padding
    elif 'face' in landmarks:
        face = landmarks['face']
        face_ys = face[:, 1]
        face_xs = face[:, 0]
        face_h = face_ys.max() - face_ys.min()
        face_center_x = (face_xs.max() + face_xs.min()) / 2

        top = face_ys.min() - face_h * 0.5
        bottom = face_ys.max() + face_h * 3.0  # estimate torso as ~3x face height
        width_est = face_h * 2.5
        left = face_center_x - width_est
        right = face_center_x + width_est
    else:
        # Fallback: center crop
        cx, cy = img_w / 2, img_h / 2
        crop_h = img_h * 0.8
        crop_w = crop_h * 9 / 16
        return (int(cx - crop_w / 2), int(cy - crop_h / 2), int(crop_w), int(crop_h))

    # Enforce 9:16 aspect ratio
    crop_h = bottom - top
    crop_w = crop_h * 9 / 16
    center_x = (left + right) / 2

    x = center_x - crop_w / 2
    y = top

    # Clamp to image bounds
    x = max(0, min(x, img_w - crop_w))
    y = max(0, min(y, img_h - crop_h))
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    return (int(x), int(y), int(crop_w), int(crop_h))


def compute_stable_crop(all_landmarks: list, img_sizes: list) -> tuple:
    """Compute a single stable crop rectangle across all photos.

    Takes the union of individual crops for consistent framing.
    """
    crops = []
    for lm, (w, h) in zip(all_landmarks, img_sizes):
        if lm is not None:
            crops.append(compute_upper_body_crop(lm, w, h))

    if not crops:
        w, h = img_sizes[0]
        crop_h = int(h * 0.8)
        crop_w = int(crop_h * 9 / 16)
        return ((w - crop_w) // 2, (h - crop_h) // 2, crop_w, crop_h)

    # Union of all crop rectangles
    min_x = min(c[0] for c in crops)
    min_y = min(c[1] for c in crops)
    max_x2 = max(c[0] + c[2] for c in crops)
    max_y2 = max(c[1] + c[3] for c in crops)

    # Re-enforce 9:16 from the union
    union_h = max_y2 - min_y
    union_w = max_x2 - min_x
    target_w = union_h * 9 / 16

    if target_w < union_w:
        # Need to expand height to fit width
        target_h = union_w * 16 / 9
        expand = (target_h - union_h) / 2
        min_y -= expand
        union_h = target_h
    else:
        # Expand width to fit height
        expand = (target_w - union_w) / 2
        min_x -= expand
        union_w = target_w

    # Clamp to first image bounds (assume all photos are similar size)
    ref_w, ref_h = img_sizes[0]
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    union_w = min(union_w, ref_w - min_x)
    union_h = min(union_h, ref_h - min_y)

    return (int(min_x), int(min_y), int(union_w), int(union_h))


def crop_and_resize(image_bgr: np.ndarray, crop_rect: tuple) -> np.ndarray:
    """Crop and resize image to OUTPUT_WIDTH x OUTPUT_HEIGHT."""
    x, y, w, h = crop_rect
    img_h, img_w = image_bgr.shape[:2]

    # Handle crops that extend beyond image boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    cropped = image_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = image_bgr  # fallback to full image

    return cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)


def select_morph_points(landmarks: dict, crop_rect: tuple) -> np.ndarray:
    """Select and remap landmark points to cropped/resized coordinate space.

    Returns Nx2 array of points including face mesh, body pose, and boundary points.
    """
    cx, cy, cw, ch = crop_rect
    scale_x = OUTPUT_WIDTH / cw
    scale_y = OUTPUT_HEIGHT / ch

    points = []

    if 'face' in landmarks:
        for px, py in landmarks['face']:
            points.append(((px - cx) * scale_x, (py - cy) * scale_y))

    if 'pose' in landmarks:
        pose = landmarks['pose']
        for idx in _UPPER_BODY_POSE_INDICES:
            px, py = pose[idx]
            points.append(((px - cx) * scale_x, (py - cy) * scale_y))

        # Add torso midpoints (between shoulder and hip on each side)
        for s_idx, h_idx in [(11, 23), (12, 24)]:
            sx, sy = pose[s_idx]
            hx, hy = pose[h_idx]
            for t in [0.25, 0.5, 0.75]:
                mx = sx + t * (hx - sx)
                my = sy + t * (hy - sy)
                points.append(((mx - cx) * scale_x, (my - cy) * scale_y))

    # Add boundary points (corners + edge midpoints)
    w, h = OUTPUT_WIDTH, OUTPUT_HEIGHT
    boundary = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1),
    ]
    points.extend(boundary)

    pts = np.array(points, dtype=np.float64)

    # Clamp to image bounds
    pts[:, 0] = np.clip(pts[:, 0], 0, OUTPUT_WIDTH - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, OUTPUT_HEIGHT - 1)

    return pts


def compute_delaunay(points: np.ndarray, width: int, height: int) -> list:
    """Compute Delaunay triangulation. Returns list of (i, j, k) index triples."""
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangles = subdiv.getTriangleList()
    result = []

    for t in triangles:
        pts_tri = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = []
        for pt in pts_tri:
            diffs = points - np.array(pt)
            dists = np.sum(diffs ** 2, axis=1)
            indices.append(int(np.argmin(dists)))

        # Skip degenerate triangles
        if len(set(indices)) == 3:
            result.append(tuple(indices))

    return result


def morph_triangle(src_img, dst_img, output, tri_src, tri_dst, tri_morph, alpha):
    """Warp and alpha-blend a single triangle from src and dst into output."""
    # Bounding rectangles
    r_src = cv2.boundingRect(np.float32([tri_src]))
    r_dst = cv2.boundingRect(np.float32([tri_dst]))
    r_morph = cv2.boundingRect(np.float32([tri_morph]))

    # Offset triangles to their bounding rects
    tri_src_offset = [(p[0] - r_src[0], p[1] - r_src[1]) for p in tri_src]
    tri_dst_offset = [(p[0] - r_dst[0], p[1] - r_dst[1]) for p in tri_dst]
    tri_morph_offset = [(p[0] - r_morph[0], p[1] - r_morph[1]) for p in tri_morph]

    # Crop source regions
    x, y, w, h = r_src
    if w <= 0 or h <= 0:
        return
    src_crop = src_img[y:y+h, x:x+w].copy()

    x, y, w, h = r_dst
    if w <= 0 or h <= 0:
        return
    dst_crop = dst_img[y:y+h, x:x+w].copy()

    x, y, w, h = r_morph
    if w <= 0 or h <= 0:
        return

    # Affine transforms
    mat_src = cv2.getAffineTransform(
        np.float32(tri_src_offset), np.float32(tri_morph_offset)
    )
    mat_dst = cv2.getAffineTransform(
        np.float32(tri_dst_offset), np.float32(tri_morph_offset)
    )

    warped_src = cv2.warpAffine(src_crop, mat_src, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
    warped_dst = cv2.warpAffine(dst_crop, mat_dst, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)

    # Create mask for the triangle
    mask = np.zeros((h, w, 3), dtype=np.float64)
    cv2.fillConvexPoly(mask, np.int32(tri_morph_offset), (1.0, 1.0, 1.0), 16, 0)

    # Blend and write to output
    blended = (1.0 - alpha) * warped_src + alpha * warped_dst
    region = output[y:y+h, x:x+w]
    region[:] = region * (1 - mask) + blended * mask


def generate_morph_frame(img_a, img_b, points_a, points_b, triangles, alpha):
    """Generate a single morph frame between two images at the given alpha."""
    # Interpolate points
    points_morph = (1.0 - alpha) * points_a + alpha * points_b

    output = np.zeros(img_a.shape, dtype=img_a.dtype)

    for i, j, k in triangles:
        tri_src = [points_a[i], points_a[j], points_a[k]]
        tri_dst = [points_b[i], points_b[j], points_b[k]]
        tri_morph = [points_morph[i], points_morph[j], points_morph[k]]

        morph_triangle(img_a, img_b, output, tri_src, tri_dst, tri_morph, alpha)

    return output


def encode_frames_to_mp4(frames: list, fps: int = FPS) -> bytes:
    """Encode frames to H.264 MP4 video bytes via imageio-ffmpeg."""
    import imageio.v3 as iio

    if not frames:
        raise RuntimeError("No frames to encode")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(tmp_fd)

    try:
        # imageio expects RGB, our frames are BGR
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        iio.imwrite(tmp_path, rgb_frames, fps=fps, codec="libx264",
                     plugin="pyav")

        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_progress_morph_video(photo_bytes_list: list) -> bytes:
    """Generate a morphing progress video from a list of JPEG photo bytes.

    Args:
        photo_bytes_list: List of JPEG bytes in chronological order (oldest first).
                         Must have at least 2 photos.

    Returns:
        MP4 video bytes.
    """
    if len(photo_bytes_list) < 2:
        raise ValueError("Need at least 2 photos to generate a morph video")

    # Decode all photos
    images = []
    for i, photo_bytes in enumerate(photo_bytes_list):
        buf = np.frombuffer(photo_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode photo {i}")
        images.append(img)

    # Detect landmarks on all photos
    all_landmarks = [detect_landmarks(img) for img in images]

    # Fallback: fill missing landmarks from nearest neighbor
    success_indices = [i for i, lm in enumerate(all_landmarks) if lm is not None]
    if len(success_indices) < 2:
        raise RuntimeError(
            "Could not detect landmarks on enough photos. "
            f"Detection succeeded on {len(success_indices)} of {len(images)} photos."
        )

    for i in range(len(all_landmarks)):
        if all_landmarks[i] is None:
            nearest = min(success_indices, key=lambda j: abs(j - i))
            all_landmarks[i] = all_landmarks[nearest]

    # Compute stable crop across all photos
    img_sizes = [(img.shape[1], img.shape[0]) for img in images]
    crop_rect = compute_stable_crop(all_landmarks, img_sizes)

    # Crop and resize all photos
    cropped_images = [crop_and_resize(img, crop_rect) for img in images]

    # Select morph points for each photo
    all_points = [select_morph_points(lm, crop_rect) for lm in all_landmarks]

    # Ensure all point arrays have the same size (they should if landmarks are consistent)
    min_points = min(len(p) for p in all_points)
    all_points = [p[:min_points] for p in all_points]

    # Generate all frames
    frames = []
    for i in range(len(cropped_images)):
        # Hold frames for current photo
        for _ in range(HOLD_FRAMES):
            frames.append(cropped_images[i])

        # Transition to next photo
        if i < len(cropped_images) - 1:
            img_a = cropped_images[i].astype(np.float64)
            img_b = cropped_images[i + 1].astype(np.float64)
            pts_a = all_points[i]
            pts_b = all_points[i + 1]

            # Compute triangulation on midpoint set for stability
            mid_points = (pts_a + pts_b) / 2.0
            triangles = compute_delaunay(mid_points, OUTPUT_WIDTH, OUTPUT_HEIGHT)

            for f in range(FRAMES_PER_TRANSITION):
                alpha = (f + 1) / (FRAMES_PER_TRANSITION + 1)
                frame = generate_morph_frame(img_a, img_b, pts_a, pts_b, triangles, alpha)
                frames.append(np.clip(frame, 0, 255).astype(np.uint8))

    return encode_frames_to_mp4(frames)
