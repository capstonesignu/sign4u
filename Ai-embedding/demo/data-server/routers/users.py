"""
users.py — 사용자 계정 관리 API

엔드포인트:
  GET    /data/users/exists      — 계정 존재 여부
  POST   /data/users/setup       — 첫 관리자 계정 생성 (테이블 비어있을 때만)
  POST   /data/users/signup      — 일반 계정 가입 (미승인 상태)
  POST   /data/users/login       — 로그인 검증 → {ok, is_admin?, reason?}
  GET    /data/users/pending     — 승인 대기 목록
  GET    /data/users/list        — 활성 계정 전체 목록
  POST   /data/users/approve     — 계정 승인
  DELETE /data/users/{user_id}   — 계정 삭제
"""
from __future__ import annotations

import asyncpg
import hashlib
import hmac
import secrets
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/data/users", tags=["users"])

_ITERATIONS = 260_000


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()
    return f"pbkdf2:{salt}:{h}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split(":")
    except ValueError:
        return False
    new_h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()
    return hmac.compare_digest(h, new_h)


async def _pool(request: Request) -> asyncpg.Pool:
    pool = getattr(request.app.state, "jbedu_pool", None)
    if pool is None:
        raise HTTPException(503, "Database not available")
    return pool


@router.get("/exists")
async def users_exist(request: Request):
    pool = await _pool(request)
    row = await pool.fetchrow("SELECT 1 FROM users LIMIT 1")
    return {"exists": row is not None}


@router.post("/setup")
async def setup_first_user(request: Request, body: dict):
    """계정이 하나도 없을 때만 첫 번째 관리자 계정 생성."""
    pool = await _pool(request)
    if await pool.fetchrow("SELECT 1 FROM users LIMIT 1"):
        raise HTTPException(403, "이미 계정이 존재합니다")

    username  = (body.get("username") or "").strip()
    full_name = (body.get("full_name") or "").strip()
    password  = body.get("password") or ""
    if not username or not full_name or len(password) < 6:
        raise HTTPException(400, "모든 항목을 올바르게 입력해주세요 (비밀번호 최소 6자)")

    try:
        await pool.execute(
            """INSERT INTO users (username, full_name, password_hash, is_active, is_admin)
               VALUES ($1, $2, $3, TRUE, TRUE)""",
            username, full_name, _hash_password(password),
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(409, "이미 사용 중인 아이디입니다")
    return {"ok": True}


@router.post("/signup")
async def create_user(request: Request, body: dict):
    """일반 계정 가입 — 미승인(is_active=FALSE) 상태로 생성."""
    pool = await _pool(request)
    username  = (body.get("username") or "").strip()
    full_name = (body.get("full_name") or "").strip()
    password  = body.get("password") or ""
    if not username or not full_name or len(password) < 6:
        raise HTTPException(400, "모든 항목을 올바르게 입력해주세요 (비밀번호 최소 6자)")

    try:
        await pool.execute(
            """INSERT INTO users (username, full_name, password_hash, is_active, is_admin)
               VALUES ($1, $2, $3, FALSE, FALSE)""",
            username, full_name, _hash_password(password),
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(409, "이미 사용 중인 아이디입니다")
    return {"ok": True}


@router.post("/login")
async def verify_login(request: Request, body: dict):
    """아이디/비밀번호 검증. ok / reason(invalid|pending) / is_admin 반환."""
    pool = await _pool(request)
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    row = await pool.fetchrow(
        "SELECT password_hash, is_active, is_admin, full_name FROM users WHERE username = $1",
        username,
    )
    if not row or not _verify_password(password, row["password_hash"]):
        return {"ok": False, "reason": "invalid"}
    if not row["is_active"]:
        return {"ok": False, "reason": "pending"}
    return {"ok": True, "is_admin": bool(row["is_admin"]), "full_name": row["full_name"] or ""}


@router.get("/pending")
async def list_pending(request: Request):
    """승인 대기 중인 계정 목록."""
    pool = await _pool(request)
    rows = await pool.fetch(
        "SELECT id, username, full_name, created_at FROM users"
        " WHERE is_active = FALSE ORDER BY created_at"
    )
    return {"users": [dict(r) for r in rows]}


@router.get("/list")
async def list_users(request: Request):
    """활성 계정 전체 목록."""
    pool = await _pool(request)
    rows = await pool.fetch(
        "SELECT id, username, full_name, is_admin, created_at FROM users"
        " WHERE is_active = TRUE ORDER BY created_at"
    )
    return {"users": [dict(r) for r in rows]}


@router.post("/approve")
async def approve_user(request: Request, body: dict):
    """계정 승인 — is_active=TRUE."""
    pool = await _pool(request)
    user_id = body.get("user_id")
    if not user_id:
        raise HTTPException(400, "user_id 필요")
    result = await pool.execute(
        "UPDATE users SET is_active = TRUE WHERE id = $1 AND is_active = FALSE",
        int(user_id),
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "해당 사용자가 없거나 이미 승인됨")
    return {"ok": True}


@router.delete("/{user_id}")
async def delete_user(request: Request, user_id: int):
    """계정 삭제 (거부 또는 강제 삭제). 관리자 본인은 삭제 불가."""
    pool = await _pool(request)
    # 마지막 관리자 삭제 방지
    admin_count = await pool.fetchval(
        "SELECT COUNT(*) FROM users WHERE is_admin = TRUE AND is_active = TRUE"
    )
    is_admin_target = await pool.fetchval(
        "SELECT is_admin FROM users WHERE id = $1", user_id
    )
    if is_admin_target and admin_count <= 1:
        raise HTTPException(400, "마지막 관리자는 삭제할 수 없습니다")

    result = await pool.execute("DELETE FROM users WHERE id = $1", user_id)
    if result == "DELETE 0":
        raise HTTPException(404, "해당 사용자를 찾을 수 없습니다")
    return {"ok": True}
