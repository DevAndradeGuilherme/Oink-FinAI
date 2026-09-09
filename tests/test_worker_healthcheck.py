from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from oink_finai.services.worker_heartbeat import WorkerHealthState


async def test_worker_healthcheck_accepts_only_its_recent_running_record(
    monkeypatch, tmp_path
) -> None:
    from oink_finai import worker_healthcheck

    worker_id = uuid4()
    id_path = tmp_path / "opaque-id"
    id_path.write_text(str(worker_id), encoding="ascii")
    settings = SimpleNamespace(
        worker_heartbeat_id_path=str(id_path),
        worker_heartbeat_stale_seconds=60,
        worker_heartbeat_database_timeout_seconds=3,
        worker_heartbeat_retention_days=7,
    )
    service = Mock(check_worker=AsyncMock(return_value=WorkerHealthState.HEALTHY))
    engine = Mock(dispose=AsyncMock())
    constructor = Mock(return_value=service)
    monkeypatch.setattr(worker_healthcheck, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_healthcheck, "WorkerHeartbeatService", constructor)
    monkeypatch.setattr(worker_healthcheck, "engine", engine)

    assert await worker_healthcheck.check()
    service.check_worker.assert_awaited_once_with(worker_id)
    engine.dispose.assert_awaited_once()


@pytest.mark.parametrize(
    "state",
    [WorkerHealthState.STALE, WorkerHealthState.ABSENT, WorkerHealthState.STOPPED],
)
async def test_worker_healthcheck_rejects_every_nonhealthy_state(
    monkeypatch, tmp_path, state: WorkerHealthState
) -> None:
    from oink_finai import worker_healthcheck

    id_path = tmp_path / "opaque-id"
    id_path.write_text(str(uuid4()), encoding="ascii")
    settings = SimpleNamespace(
        worker_heartbeat_id_path=str(id_path),
        worker_heartbeat_stale_seconds=60,
        worker_heartbeat_database_timeout_seconds=3,
        worker_heartbeat_retention_days=7,
    )
    service = Mock(check_worker=AsyncMock(return_value=state))
    monkeypatch.setattr(worker_healthcheck, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_healthcheck, "WorkerHeartbeatService", Mock(return_value=service))
    monkeypatch.setattr(worker_healthcheck, "engine", Mock(dispose=AsyncMock()))

    assert not await worker_healthcheck.check()


async def test_worker_healthcheck_fails_closed_without_exposing_database_error(
    monkeypatch, tmp_path, capsys
) -> None:
    from oink_finai import worker_healthcheck

    id_path = tmp_path / "opaque-id"
    id_path.write_text(str(uuid4()), encoding="ascii")
    settings = SimpleNamespace(
        worker_heartbeat_id_path=str(id_path),
        worker_heartbeat_stale_seconds=60,
        worker_heartbeat_database_timeout_seconds=3,
        worker_heartbeat_retention_days=7,
    )
    private_error = "postgresql://private-user:private-password@private-host/private-db"
    service = Mock(check_worker=AsyncMock(side_effect=RuntimeError(private_error)))
    monkeypatch.setattr(worker_healthcheck, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_healthcheck, "WorkerHeartbeatService", Mock(return_value=service))
    monkeypatch.setattr(worker_healthcheck, "engine", Mock(dispose=AsyncMock()))

    assert not await worker_healthcheck.check()
    captured = capsys.readouterr()
    assert private_error not in captured.out + captured.err


async def test_worker_healthcheck_rejects_missing_or_invalid_id_before_database(
    monkeypatch, tmp_path
) -> None:
    from oink_finai import worker_healthcheck

    id_path = tmp_path / "opaque-id"
    id_path.write_text("not-a-uuid", encoding="ascii")
    settings = SimpleNamespace(worker_heartbeat_id_path=str(id_path))
    constructor = Mock()
    monkeypatch.setattr(worker_healthcheck, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_healthcheck, "WorkerHeartbeatService", constructor)

    assert not await worker_healthcheck.check()
    constructor.assert_not_called()
