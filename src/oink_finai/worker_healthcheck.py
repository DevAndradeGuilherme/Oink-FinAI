import asyncio
from pathlib import Path
from uuid import UUID

from oink_finai.config.settings import get_settings
from oink_finai.database.session import SessionFactory, engine
from oink_finai.services.worker_heartbeat import WorkerHealthState, WorkerHeartbeatService


async def check() -> bool:
    settings = get_settings()
    try:
        worker_id = UUID(
            Path(settings.worker_heartbeat_id_path).read_text(encoding="ascii").strip()
        )
    except (OSError, UnicodeError, ValueError):
        return False
    service = WorkerHeartbeatService(
        SessionFactory,
        stale_seconds=settings.worker_heartbeat_stale_seconds,
        database_timeout_seconds=settings.worker_heartbeat_database_timeout_seconds,
        retention_days=settings.worker_heartbeat_retention_days,
    )
    try:
        return await service.check_worker(worker_id) is WorkerHealthState.HEALTHY
    except Exception:
        return False
    finally:
        await engine.dispose()


def main() -> None:
    raise SystemExit(0 if asyncio.run(check()) else 1)


if __name__ == "__main__":
    main()
