import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services import storage

log = logging.getLogger("vocal-studio.storage")


async def _cleanup_expired_files() -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            removed = await asyncio.to_thread(storage.cleanup_old_files)
        except Exception:
            log.exception("Periodic temporary-file cleanup failed")
        else:
            if removed:
                log.info("Removed %s expired temporary storage entries", removed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.ensure_dirs()
    storage.cleanup_old_files()
    cleanup_task = asyncio.create_task(_cleanup_expired_files())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Vocal Studio API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
