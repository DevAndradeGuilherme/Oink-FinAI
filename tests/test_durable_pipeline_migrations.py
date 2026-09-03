import re
from pathlib import Path

from oink_finai.domain.enums import OutboundMessageStatus, ProcessedMessageStatus

MIGRATIONS = Path(__file__).parents[1] / "migrations" / "versions"


def test_clean_migration_enum_matches_outbound_status() -> None:
    source = (MIGRATIONS / "20260901_0002_durable_expense_pipeline.py").read_text()
    enum_block = source.split("outbound_status = postgresql.ENUM(", maxsplit=1)[1].split(
        ")", maxsplit=1
    )[0]
    values = re.findall(r'^\s+"([A-Z]+)",$', enum_block, flags=re.MULTILINE)

    assert values == [status.value for status in OutboundMessageStatus]


def test_clean_migration_has_claim_columns_and_safe_historical_backfill() -> None:
    source = (MIGRATIONS / "20260901_0002_durable_expense_pipeline.py").read_text()

    assert 'sa.Column("claimed_at", sa.DateTime(timezone=True))' in source
    assert 'sa.Column("sending_at", sa.DateTime(timezone=True))' in source
    assert 'sa.Column("claim_token", sa.Uuid())' in source
    assert 'sa.Column("status", processed_status, nullable=True)' in source
    assert "status = 'PROCESSED'" in source
    assert 'alter_column("processed_messages", "status", nullable=False)' in source
    assert (
        'server_default="PENDING"' not in source.split('sa.Column("attempt_count"', maxsplit=1)[0]
    )
    assert ProcessedMessageStatus.PROCESSED.value == "PROCESSED"


def test_corrective_migration_preserves_legitimate_pending_messages() -> None:
    source = (MIGRATIONS / "20260901_0004_reconcile_durable_pipeline.py").read_text()

    assert "ADD VALUE IF NOT EXISTS 'CLAIMED'" in source
    assert source.count("ADD COLUMN IF NOT EXISTS") == 3
    assert "status = 'PENDING' AND accepted_text = ''" in source
    assert "accepted_text != ''" not in source


def test_durable_retry_migration_has_safe_backfill_and_terminal_kind() -> None:
    source = (MIGRATIONS / "20260902_0005_durable_gemini_retries.py").read_text()

    assert "ADD VALUE IF NOT EXISTS 'PROCESSING_FAILURE'" in source
    assert 'sa.Column("processing_attempts",' in source
    assert 'sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True)' in source
    assert 'sa.Column("last_error_code", sa.String(64), nullable=True)' in source
    assert 'server_default="0"' in source
    assert (
        'alter_column("processed_messages", "processing_attempts", server_default=None)' in source
    )
    assert "UPDATE processed_messages SET last_error_code = error_code" in source
