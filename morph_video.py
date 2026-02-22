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


def compute_body_alignment(landmarks: dict, reference_landmarks: dict,
                           img_w: int = 0, img_h: int = 0) -> np.ndarray:
    """Compute similarity transform to align body to a reference pose.

    Uses shoulder and hip landmarks as stable torso anchors.
    Clamps rotation, scale, and translation to prevent aggressive warping.
    Returns a 2x3 affine matrix for cv2.warpAffine.
    """
    if 'pose' not in landmarks or 'pose' not in reference_landmarks:
        return np.eye(2, 3, dtype=np.float64)

    pose = landmarks['pose']
    ref_pose = reference_landmarks['pose']

    # Torso anchors: shoulders (11, 12) and hips (23, 24)
    src_points = np.array([
        pose[11], pose[12], pose[23], pose[24],
    ], dtype=np.float64)
    dst_points = np.array([
        ref_pose[11], ref_pose[12], ref_pose[23], ref_pose[24],
    ], dtype=np.float64)

    # Similarity transform (translate + rotate + uniform scale, no shear)
    transform, _ = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS)
    if transform is None:
        return np.eye(2, 3, dtype=np.float64)

    # Decompose: [[s*cos(θ), -s*sin(θ), tx], [s*sin(θ), s*cos(θ), ty]]
    a, b, tx = transform[0]
    c, d, ty = transform[1]  # c = s*sin(θ), d = s*cos(θ)
    scale = np.sqrt(a * a + b * b)
    angle = np.arctan2(c, a)  # radians

    # Clamp: rotation ±10°, scale 0.65–1.5, translation ±15% of image dims
    max_angle = np.radians(10)
    angle = np.clip(angle, -max_angle, max_angle)
    scale = np.clip(scale, 0.65, 1.5)

    if img_w > 0 and img_h > 0:
        max_tx = img_w * 0.15
        max_ty = img_h * 0.15
        tx = np.clip(tx, -max_tx, max_tx)
        ty = np.clip(ty, -max_ty, max_ty)

    # Rebuild clamped transform
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty],
    ], dtype=np.float64)


def align_image(image_bgr: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Warp image using an affine transform matrix."""
    h, w = image_bgr.shape[:2]
    return cv2.warpAffine(image_bgr, transform, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def transform_landmarks(landmarks: dict, transform: np.ndarray) -> dict:
    """Apply affine transform to all landmark coordinates."""
    result = {}
    for key in landmarks:
        pts = landmarks[key]
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])  # Nx3
        transformed = (transform @ pts_h.T).T  # Nx2
        result[key] = transformed
    return result


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE to luminance channel only (preserves color)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def reinhard_color_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Transfer color distribution from reference to source using Reinhard's method.

    Matches mean and std of each LAB channel to normalize lighting and color temperature.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float64)

    src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1))
    ref_mean, ref_std = ref_lab.mean(axis=(0, 1)), ref_lab.std(axis=(0, 1))

    # Avoid division by zero
    src_std = np.where(src_std == 0, 1, src_std)

    result = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def normalize_lighting(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Full lighting normalization: CLAHE then Reinhard color transfer."""
    enhanced = apply_clahe(source)
    return reinhard_color_transfer(enhanced, reference)


def smoothstep(t: float) -> float:
    """Hermite smoothstep: slow start, fast middle, slow end."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def normalize_resolutions(images: list) -> list:
    """Resize all images to match the first image's resolution."""
    ref_h, ref_w = images[0].shape[:2]
    result = [images[0]]
    for img in images[1:]:
        h, w = img.shape[:2]
        if w != ref_w or h != ref_h:
            result.append(cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA))
        else:
            result.append(img)
    return result


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

        # Vertical: from above head to hips, extra top padding for wider 9:16 frame
        mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        head_margin = (mid_shoulder_y - nose[1]) * 2.5
        top = nose[1] - head_margin
        bottom = max(l_hip[1], r_hip[1])

        # Horizontal: from outermost arm landmarks with padding
        arm_xs = [l_shoulder[0], r_shoulder[0],
                  pose[13][0], pose[14][0],   # elbows
                  pose[15][0], pose[16][0]]    # wrists
        left = min(arm_xs)
        right = max(arm_xs)
        shoulder_width = max(l_shoulder[0], r_shoulder[0]) - min(l_shoulder[0], r_shoulder[0])
        padding = shoulder_width * 0.7
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

    # Enforce 9:16 aspect ratio (height drives width)
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

    # Clamp to the smallest image bounds across all photos
    smallest_w = min(w for w, h in img_sizes)
    smallest_h = min(h for w, h in img_sizes)
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    union_w = min(union_w, smallest_w - min_x)
    union_h = min(union_h, smallest_h - min_y)

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

    Pipeline:
        1. Decode photos
        2. Detect landmarks
        3. Align bodies to reference pose (similarity transform)
        4. Normalize lighting (CLAHE + Reinhard color transfer)
        5. Compute stable crop and resize
        6. Generate morph frames with smoothstep interpolation

    Args:
        photo_bytes_list: List of JPEG bytes in chronological order (oldest first).
                         Must have at least 2 photos.

    Returns:
        MP4 video bytes.
    """
    if len(photo_bytes_list) < 2:
        raise ValueError("Need at least 2 photos to generate a morph video")

    # 1. Decode all photos
    images = []
    for i, photo_bytes in enumerate(photo_bytes_list):
        buf = np.frombuffer(photo_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode photo {i}")
        images.append(img)

    # 2. Normalize all photos to same resolution
    images = normalize_resolutions(images)

    # 3. Detect landmarks on all photos
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

    # 4. Align bodies to reference pose (first photo with landmarks)
    ref_idx = success_indices[0]
    ref_landmarks = all_landmarks[ref_idx]
    for i in range(len(images)):
        if i == ref_idx:
            continue
        h, w = images[i].shape[:2]
        transform = compute_body_alignment(all_landmarks[i], ref_landmarks, w, h)
        images[i] = align_image(images[i], transform)
        all_landmarks[i] = transform_landmarks(all_landmarks[i], transform)

    # 5. Normalize lighting to reference photo
    ref_image = apply_clahe(images[ref_idx])
    for i in range(len(images)):
        if i == ref_idx:
            images[i] = ref_image
        else:
            images[i] = normalize_lighting(images[i], ref_image)

    # 6. Compute stable crop across all aligned/normalized photos
    img_sizes = [(img.shape[1], img.shape[0]) for img in images]
    crop_rect = compute_stable_crop(all_landmarks, img_sizes)

    # Crop and resize all photos
    cropped_images = [crop_and_resize(img, crop_rect) for img in images]

    # Select morph points for each photo
    all_points = [select_morph_points(lm, crop_rect) for lm in all_landmarks]

    # Ensure all point arrays have the same size
    min_points = min(len(p) for p in all_points)
    all_points = [p[:min_points] for p in all_points]

    # 7. Generate all frames with smoothstep interpolation
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
                linear_t = (f + 1) / (FRAMES_PER_TRANSITION + 1)
                alpha = smoothstep(linear_t)
                frame = generate_morph_frame(img_a, img_b, pts_a, pts_b, triangles, alpha)
                frames.append(np.clip(frame, 0, 255).astype(np.uint8))

    return encode_frames_to_mp4(frames)


# --- Debug visualization ---

_POSE_LABELS = {
    0: "nose", 11: "L.shldr", 12: "R.shldr",
    13: "L.elbow", 14: "R.elbow", 15: "L.wrist", 16: "R.wrist",
    23: "L.hip", 24: "R.hip",
}
_SKELETON_EDGES = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
    (23, 24),            # hips
    (11, 23), (12, 24),  # torso sides
]


def draw_debug_overlay(image: np.ndarray, landmarks: dict,
                       crop_rect: tuple, alignment_info: dict = None) -> np.ndarray:
    """Draw landmarks, skeleton, crop rect, and alignment info on image copy."""
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # Face mesh — small cyan dots
    if 'face' in landmarks:
        for px, py in landmarks['face']:
            x, y = int(px), int(py)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), 2, (255, 255, 0), -1)

    # Pose skeleton — green lines
    if 'pose' in landmarks:
        pose = landmarks['pose']
        for i1, i2 in _SKELETON_EDGES:
            if i1 < len(pose) and i2 < len(pose):
                pt1 = (int(pose[i1][0]), int(pose[i1][1]))
                pt2 = (int(pose[i2][0]), int(pose[i2][1]))
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

        # Pose landmarks — colored circles with labels
        for idx, label in _POSE_LABELS.items():
            if idx < len(pose):
                x, y = int(pose[idx][0]), int(pose[idx][1])
                cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(canvas, label, (x + 8, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Crop rectangle — bright magenta
    cx, cy, cw, ch = crop_rect
    cv2.rectangle(canvas, (int(cx), int(cy)),
                  (int(cx + cw), int(cy + ch)), (255, 0, 255), 3)

    # Alignment info — text in top-left
    if alignment_info:
        lines = [
            f"scale: {alignment_info.get('scale', 0):.3f}",
            f"rot: {alignment_info.get('angle_deg', 0):.1f} deg",
            f"tx: {alignment_info.get('tx', 0):.1f}  ty: {alignment_info.get('ty', 0):.1f}",
        ]
        for i, line in enumerate(lines):
            y_pos = 30 + i * 25
            cv2.putText(canvas, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return canvas


def generate_debug_images(photo_bytes_list: list) -> list[bytes]:
    """Run the morph pipeline and return annotated JPEG bytes for each photo.

    Draws landmarks, skeleton, crop rectangle, and alignment info on each image.
    Returns list of JPEG bytes in same order as input.
    """
    if len(photo_bytes_list) < 2:
        raise ValueError("Need at least 2 photos")

    # 1. Decode
    images = []
    for i, photo_bytes in enumerate(photo_bytes_list):
        buf = np.frombuffer(photo_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode photo {i}")
        images.append(img)

    # 2. Normalize resolutions
    images = normalize_resolutions(images)

    # 3. Detect landmarks
    all_landmarks = [detect_landmarks(img) for img in images]

    success_indices = [i for i, lm in enumerate(all_landmarks) if lm is not None]
    if len(success_indices) < 2:
        raise RuntimeError(
            f"Landmarks detected on only {len(success_indices)} of {len(images)} photos."
        )
    for i in range(len(all_landmarks)):
        if all_landmarks[i] is None:
            nearest = min(success_indices, key=lambda j: abs(j - i))
            all_landmarks[i] = all_landmarks[nearest]

    # 4. Align bodies to reference pose
    ref_idx = success_indices[0]
    ref_landmarks = all_landmarks[ref_idx]
    alignment_infos = [{'scale': 1.0, 'angle_deg': 0.0, 'tx': 0.0, 'ty': 0.0}
                       for _ in images]

    for i in range(len(images)):
        if i == ref_idx:
            continue
        ih, iw = images[i].shape[:2]
        transform = compute_body_alignment(all_landmarks[i], ref_landmarks, iw, ih)
        # Extract alignment info for display
        a, b, tx = transform[0]
        c, d, ty = transform[1]
        scale = np.sqrt(a * a + b * b)
        angle = np.degrees(np.arctan2(c, a))
        alignment_infos[i] = {
            'scale': float(scale), 'angle_deg': float(angle),
            'tx': float(tx), 'ty': float(ty),
        }
        images[i] = align_image(images[i], transform)
        all_landmarks[i] = transform_landmarks(all_landmarks[i], transform)

    # 5. Compute stable crop (same as video pipeline)
    img_sizes = [(img.shape[1], img.shape[0]) for img in images]
    crop_rect = compute_stable_crop(all_landmarks, img_sizes)

    # 6. Draw debug overlay on each image
    result = []
    for i in range(len(images)):
        debug_img = draw_debug_overlay(images[i], all_landmarks[i],
                                       crop_rect, alignment_infos[i])
        _, jpeg_buf = cv2.imencode('.jpg', debug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result.append(jpeg_buf.tobytes())

    return result
