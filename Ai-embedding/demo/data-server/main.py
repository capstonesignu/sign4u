import os
from contextlib import asynccontextmanager

import asyncpg
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from routers.jbedu import router as jbedu_router
from routers.keypoints import router as keypoints_router
from routers.users import router as users_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")
    print("[startup] Connecting to PostgreSQL...")
    pool = await asyncpg.create_pool(config.DATABASE_URL, min_size=2, max_size=10)
    app.state.jbedu_pool = pool

    await pool.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            username      VARCHAR(64) UNIQUE NOT NULL,
            full_name     VARCHAR(100) NOT NULL DEFAULT '',
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMPTZ DEFAULT NOW(),
            is_active     BOOLEAN NOT NULL DEFAULT FALSE,
            is_admin      BOOLEAN NOT NULL DEFAULT FALSE
        )
    """)
    # 기존 테이블 마이그레이션 (멱등)
    await pool.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(100) NOT NULL DEFAULT ''"
    )
    await pool.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE"
    )
    # 관리자가 한 명도 없으면 가장 먼저 만들어진 활성 계정을 관리자로 승격
    admin_count = await pool.fetchval(
        "SELECT COUNT(*) FROM users WHERE is_admin = TRUE AND is_active = TRUE"
    )
    if admin_count == 0:
        await pool.execute(
            "UPDATE users SET is_admin = TRUE"
            " WHERE id = (SELECT MIN(id) FROM users WHERE is_active = TRUE)"
        )
        print("[startup] 기존 계정을 관리자로 승격했습니다")
    print("[startup] Ready")
    yield
    await pool.close()


app = FastAPI(title="KSL jbedu Data API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(keypoints_router)  # specific /keypoints/list, /keypoints/download 먼저
app.include_router(jbedu_router)       # /keypoints/{uuid} catch-all은 나중에
app.include_router(users_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
