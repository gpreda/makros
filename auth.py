"""Google OAuth authentication for makros."""

import os

import psycopg2
from psycopg2.extras import RealDictCursor
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse


GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

router = APIRouter(tags=["auth"])


def get_or_create_user(db_url: str, google_id: str, email: str, name: str) -> int:
    """Find or create a user by Google ID, falling back to email match for the seed user.

    On first login, if a user with this email exists but has no google_id yet
    (the seeded user), their google_id is filled in so all existing data stays
    associated with user id=1.

    Returns the user's integer id.
    """
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Match by google_id (subsequent logins)
            cur.execute("SELECT id FROM users WHERE google_id = %s", (google_id,))
            user = cur.fetchone()
            if user:
                cur.execute(
                    "UPDATE users SET email = %s, name = %s WHERE id = %s",
                    (email, name, user['id'])
                )
                conn.commit()
                return user['id']

            # 2. Match by email with no google_id (first login of seeded user)
            cur.execute(
                "SELECT id FROM users WHERE email = %s AND google_id IS NULL",
                (email,)
            )
            user = cur.fetchone()
            if user:
                cur.execute(
                    "UPDATE users SET google_id = %s, name = %s WHERE id = %s",
                    (google_id, name, user['id'])
                )
                conn.commit()
                return user['id']

            # 3. New user
            cur.execute(
                "INSERT INTO users (google_id, email, name) VALUES (%s, %s, %s) RETURNING id",
                (google_id, email, name)
            )
            user_id = cur.fetchone()['id']
        conn.commit()
        return user_id
    finally:
        conn.close()


@router.get("/login")
async def login(request: Request):
    """Redirect to Google OAuth login."""
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    """Handle Google OAuth callback, create/update user, set session."""
    token = await oauth.google.authorize_access_token(request)
    userinfo = token["userinfo"]
    db_url = os.environ.get('DATABASE_URL', 'postgresql://predator@localhost:5432/makros')
    user_id = get_or_create_user(
        db_url,
        google_id=userinfo["sub"],
        email=userinfo["email"],
        name=userinfo.get("name", userinfo["email"]),
    )
    request.session["user_id"] = user_id
    return RedirectResponse(url="/", status_code=303)


@router.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to login."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)
