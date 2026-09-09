"""Destructive least-privilege checks against an isolated temporary PostgreSQL container."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

MARKER = "I_UNDERSTAND_ONLY_A_DISPOSABLE_POSTGRES_WILL_BE_DESTROYED"
ROOT = Path(__file__).resolve().parents[2]


class CheckFailed(RuntimeError):
    pass


def run(
    arguments: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    stdout=None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        arguments,
        check=False,
        env=env,
        input=input_text,
        text=True,
        stdout=stdout if stdout is not None else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        diagnostic = (result.stdout or "") + (result.stderr or "")
        diagnostic = re.sub(r"postgresql(?:\+asyncpg)?://\S+", "<redacted-url>", diagnostic)
        diagnostic = diagnostic.strip()
        raise CheckFailed(
            f"command failed without exposing its arguments (exit {result.returncode}):\n"
            f"{diagnostic}"
        )
    return result


def docker(*arguments: str, **kwargs) -> subprocess.CompletedProcess[str]:
    return run(["docker", *arguments], **kwargs)


def write_secret(path: Path, value: str) -> None:
    path.write_text(value + "\n", encoding="utf-8", newline="\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def connection_url(role: str, password: str, database: str) -> str:
    return (
        f"postgresql+asyncpg://{quote(role, safe='')}:{quote(password, safe='')}@"
        f"postgres:5432/{quote(database, safe='')}"
    )


def main() -> int:
    if os.environ.get("OINK_DESTRUCTIVE_POSTGRES_TEST_MARKER") != MARKER:
        raise CheckFailed("explicit disposable-database marker is required")
    app_image = os.environ.get("OINK_TEST_APP_IMAGE")
    if not app_image:
        raise CheckFailed("OINK_TEST_APP_IMAGE is required")
    postgres_image = os.environ.get("OINK_TEST_POSTGRES_IMAGE", "postgres:16-alpine")

    suffix = secrets.token_hex(5)
    prefix = f"oink_lp_{suffix}"
    network = f"{prefix}_network"
    volume = f"{prefix}_volume"
    postgres = f"{prefix}_postgres"
    api = f"{prefix}_api"
    worker = f"{prefix}_worker"
    database = f"{prefix}_new"
    existing_database = f"{prefix}_existing"
    bootstrap = f"{prefix}_bootstrap"
    migrator = f"{prefix}_migrator"
    runtime = f"{prefix}_runtime"
    backup = f"{prefix}_backup"
    existing_migrator = f"{prefix}_existing_migrator"
    existing_runtime = f"{prefix}_existing_runtime"
    existing_backup = f"{prefix}_existing_backup"
    legacy = f"{prefix}_legacy"
    passwords = {
        name: secrets.token_urlsafe(30)
        for name in (
            bootstrap,
            migrator,
            runtime,
            backup,
            existing_migrator,
            existing_runtime,
            existing_backup,
            legacy,
        )
    }
    rotated_passwords = {name: secrets.token_urlsafe(30) for name in passwords}

    created_network = False
    created_volume = False
    with tempfile.TemporaryDirectory(prefix="oink-lp-") as temporary:
        temp = Path(temporary)
        inventory = temp / "inventory"
        inventory.mkdir()
        for role, password in passwords.items():
            write_secret(temp / f"{role}.secret", password)
            write_secret(temp / f"{role}.rotated.secret", rotated_passwords[role])

        try:
            print("Creating isolated PostgreSQL resources...")
            docker("network", "create", "--internal", network)
            created_network = True
            docker("volume", "create", volume)
            created_volume = True
            docker(
                "run",
                "--detach",
                "--name",
                postgres,
                "--network",
                network,
                "--network-alias",
                "postgres",
                "--mount",
                f"type=volume,source={volume},target=/var/lib/postgresql/data",
                "--mount",
                f"type=bind,source={temp},target=/run/oink-secrets,readonly",
                "--mount",
                f"type=bind,source={inventory},target=/run/oink-inventory",
                "--mount",
                (
                    f"type=bind,source={ROOT / 'scripts' / 'postgres'},"
                    "target=/opt/oink/postgres,readonly"
                ),
                "--mount",
                (
                    "type=bind,source="
                    f"{ROOT / 'scripts' / 'postgres' / '001-cluster-baseline.sql'},"
                    "target=/docker-entrypoint-initdb.d/001-cluster-baseline.sql,readonly"
                ),
                "--env",
                f"POSTGRES_DB={database}",
                "--env",
                f"POSTGRES_USER={bootstrap}",
                "--env",
                f"POSTGRES_PASSWORD_FILE=/run/oink-secrets/{bootstrap}.secret",
                "--env",
                "POSTGRES_INITDB_ARGS=--auth-local=scram-sha-256 --auth-host=scram-sha-256",
                postgres_image,
            )
            print("Waiting for isolated PostgreSQL...")
            wait_for_postgres(postgres, bootstrap, database)

            print("Testing new installation and idempotence...")
            provision(
                postgres,
                "install-new.sh",
                database,
                bootstrap,
                migrator,
                runtime,
                backup,
                temp,
            )
            provision(
                postgres,
                "install-new.sh",
                database,
                bootstrap,
                migrator,
                runtime,
                backup,
                temp,
            )

            migration_env = temp / "migration.env"
            migration_url = connection_url(migrator, passwords[migrator], database)
            migration_env.write_text(
                f"APP_ENV=production\nMIGRATION_DATABASE_URL={migration_url}\n",
                encoding="utf-8",
                newline="\n",
            )
            print("Testing Alembic with migrator only...")
            run_alembic(app_image, network, migration_env, "upgrade", "head")
            for command in (("current",), ("history",), ("check",)):
                run_alembic(app_image, network, migration_env, *command)

            runtime_migration_env = temp / "runtime-migration.env"
            runtime_migration_env.write_text(
                "APP_ENV=production\n"
                f"MIGRATION_DATABASE_URL={connection_url(runtime, passwords[runtime], database)}\n",
                encoding="utf-8",
                newline="\n",
            )
            assert_runtime_cannot_migrate(app_image, network, runtime_migration_env)
            run_alembic(app_image, network, migration_env, "current", "--check-heads")

            runtime_env = temp / "runtime.env"
            runtime_env.write_text(
                runtime_environment(connection_url(runtime, passwords[runtime], database)),
                encoding="utf-8",
                newline="\n",
            )
            print("Testing API, worker, grants, denials, defaults, and pg_dump...")
            start_runtime(app_image, network, api, worker, runtime_env)
            check_runtime_processes(api, worker)

            exercise_runtime_dml(postgres, database, runtime, temp, suffix)
            exercise_denials(postgres, database, bootstrap, migrator, runtime, backup, temp, suffix)
            exercise_default_privileges(postgres, database, migrator, runtime, backup, temp, suffix)
            exercise_dump(postgres, database, backup, temp, suffix)
            exercise_usage_concurrency(app_image, network, runtime_env)
            validate(postgres, database, bootstrap, migrator, runtime, backup, temp)

            print("Testing existing-database adaptation and idempotence...")
            exercise_existing_upgrade(
                postgres=postgres,
                app_image=app_image,
                network=network,
                database=existing_database,
                bootstrap=bootstrap,
                legacy=legacy,
                migrator=existing_migrator,
                runtime=existing_runtime,
                backup=existing_backup,
                passwords=passwords,
                temp=temp,
                suffix=suffix,
            )
            validate(postgres, database, bootstrap, migrator, runtime, backup, temp)
            exercise_password_rotation(
                postgres,
                database,
                bootstrap,
                migrator,
                runtime,
                backup,
                temp,
                rotated_passwords,
            )
            print("Temporary PostgreSQL least-privilege scenarios passed.")
            return 0
        finally:
            for container in (api, worker, postgres):
                if container.startswith(prefix):
                    docker("rm", "--force", container, check=False)
            if created_volume and volume.startswith(prefix):
                docker("volume", "rm", "--force", volume, check=False)
            if created_network and network.startswith(prefix):
                docker("network", "rm", network, check=False)


def wait_for_postgres(container: str, user: str, database: str) -> None:
    for _ in range(60):
        result = docker(
            "exec",
            container,
            "pg_isready",
            "--host",
            "localhost",
            "--username",
            user,
            "--dbname",
            database,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(0.5)
    logs = docker("logs", "--tail", "30", container, check=False)
    diagnostic = (logs.stdout + logs.stderr).strip()
    raise CheckFailed(f"temporary PostgreSQL did not become ready:\n{diagnostic}")


def admin_environment(
    database: str,
    bootstrap: str,
    migrator: str,
    runtime: str,
    backup: str,
    temp: Path,
) -> dict[str, str]:
    return {
        "APP_DATABASE": database,
        "APP_SCHEMA": "public",
        "BOOTSTRAP_HOST": "localhost",
        "BOOTSTRAP_PORT": "5432",
        "BOOTSTRAP_USER": bootstrap,
        "BOOTSTRAP_PASSWORD_FILE": f"/run/oink-secrets/{bootstrap}.secret",
        "MIGRATOR_ROLE": migrator,
        "RUNTIME_ROLE": runtime,
        "BACKUP_ROLE": backup,
        "MIGRATOR_PASSWORD_FILE": f"/run/oink-secrets/{migrator}.secret",
        "RUNTIME_PASSWORD_FILE": f"/run/oink-secrets/{runtime}.secret",
        "BACKUP_PASSWORD_FILE": f"/run/oink-secrets/{backup}.secret",
        "INVENTORY_DIRECTORY": "/run/oink-inventory",
    }


def provision(
    container: str,
    script: str,
    database: str,
    bootstrap: str,
    migrator: str,
    runtime: str,
    backup: str,
    temp: Path,
    extra: dict[str, str] | None = None,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    values = admin_environment(database, bootstrap, migrator, runtime, backup, temp)
    values.update(extra or {})
    arguments = ["exec"]
    for key, value in values.items():
        arguments.extend(("--env", f"{key}={value}"))
    arguments.extend((container, "sh", f"/opt/oink/postgres/{script}"))
    return docker(*arguments, check=check)


def run_alembic(image: str, network: str, env_file: Path, *arguments: str) -> None:
    docker(
        "run",
        "--rm",
        "--network",
        network,
        "--env-file",
        str(env_file),
        "--mount",
        f"type=bind,source={ROOT / 'migrations'},target=/app/migrations,readonly",
        "--mount",
        f"type=bind,source={ROOT / 'src'},target=/app/src,readonly",
        image,
        "alembic",
        *arguments,
    )


def exercise_usage_concurrency(image: str, network: str, env_file: Path) -> None:
    docker(
        "run",
        "--rm",
        "--network",
        network,
        "--env-file",
        str(env_file),
        "--env",
        f"OINK_DESTRUCTIVE_POSTGRES_TEST_MARKER={MARKER}",
        "--mount",
        f"type=bind,source={ROOT / 'src'},target=/app/src,readonly",
        "--mount",
        (
            f"type=bind,source={ROOT / 'scripts' / 'postgres' / 'check-usage-concurrency.py'},"
            "target=/tmp/check-usage-concurrency.py,readonly"
        ),
        image,
        "python",
        "/tmp/check-usage-concurrency.py",
    )


def assert_runtime_cannot_migrate(image: str, network: str, env_file: Path) -> None:
    result = docker(
        "run",
        "--rm",
        "--network",
        network,
        "--env-file",
        str(env_file),
        "--mount",
        f"type=bind,source={ROOT / 'migrations'},target=/app/migrations,readonly",
        "--mount",
        f"type=bind,source={ROOT / 'src'},target=/app/src,readonly",
        image,
        "alembic",
        "downgrade",
        "-1",
        check=False,
    )
    if result.returncode == 0:
        raise CheckFailed("runtime unexpectedly executed an Alembic schema change")


def runtime_environment(url: str) -> str:
    return "\n".join(
        (
            "APP_ENV=production",
            "APP_DEBUG=false",
            "APP_RELOAD=false",
            f"DATABASE_URL={url}",
            "OPENAI_API_KEY=sk-synthetic-runtime-validation-0123456789",
            "EVOLUTION_BASE_URL=https://synthetic.invalid",
            "EVOLUTION_API_KEY=synthetic-evolution-key-0123456789",
            "EVOLUTION_INSTANCE=synthetic-instance",
            "EVOLUTION_WEBHOOK_SECRET=synthetic-webhook-secret-0123456789abcdef",
            "WHATSAPP_ALLOWED_NUMBERS=5511999999999",
            "WORKER_POLL_INTERVAL_SECONDS=1",
            "",
        )
    )


def start_runtime(image: str, network: str, api: str, worker: str, env_file: Path) -> None:
    docker(
        "run",
        "--detach",
        "--name",
        api,
        "--network",
        network,
        "--env-file",
        str(env_file),
        "--mount",
        f"type=bind,source={ROOT / 'src'},target=/app/src,readonly",
        image,
    )
    docker(
        "run",
        "--detach",
        "--name",
        worker,
        "--network",
        network,
        "--env-file",
        str(env_file),
        "--mount",
        f"type=bind,source={ROOT / 'src'},target=/app/src,readonly",
        image,
        "python",
        "-m",
        "oink_finai.worker",
    )


def check_runtime_processes(api: str, worker: str) -> None:
    time.sleep(2)
    for container in (api, worker):
        state = docker("inspect", "--format", "{{.State.Running}}", container).stdout.strip()
        if state != "true":
            raise CheckFailed("a runtime process failed to start")


def psql(
    container: str,
    database: str,
    role: str,
    temp: Path,
    sql: str,
    *,
    check: bool = True,
    tuples: bool = False,
) -> subprocess.CompletedProcess[str]:
    arguments = [
        "exec",
        "--env",
        f"ROLE_PASSWORD_FILE=/run/oink-secrets/{role}.secret",
        "--env",
        f"PGUSER={role}",
        "--env",
        f"PGDATABASE={database}",
        container,
        "sh",
        "/opt/oink/postgres/with-pgpass.sh",
        "psql",
        "--no-password",
        "--no-psqlrc",
        "--set=ON_ERROR_STOP=1",
    ]
    if tuples:
        arguments.extend(("--tuples-only", "--no-align"))
    arguments.extend(("--command", sql))
    return docker(*arguments, check=check)


def exercise_runtime_dml(
    container: str, database: str, runtime: str, temp: Path, suffix: str
) -> None:
    user_id = f"00000000-0000-4000-8000-{suffix.rjust(12, '0')}"
    message_id = f"10000000-0000-4000-8000-{suffix.rjust(12, '0')}"
    outbound_id = f"20000000-0000-4000-8000-{suffix.rjust(12, '0')}"
    sql = f"""
        BEGIN;
        INSERT INTO users (id, phone_number) VALUES ('{user_id}', 'synthetic-{suffix}');
        INSERT INTO processed_messages
            (id, provider, instance_id, external_message_id, user_id, accepted_text,
             message_timestamp, status, available_at, processing_attempts, source_type)
        VALUES
            ('{message_id}', 'synthetic', '{suffix}', '{suffix}', '{user_id}', 'synthetic',
             now(), 'PENDING', now(), 0, 'TEXT');
        INSERT INTO outbound_messages
            (id, user_id, processed_message_id, destination, content, content_type, kind,
             dedup_key, status, available_at, attempt_count)
        VALUES
            ('{outbound_id}', '{user_id}', '{message_id}', 'synthetic', 'synthetic', 'TEXT',
             'EXPENSE_CONFIRMATION', 'synthetic-{suffix}', 'PENDING', now(), 0);
        SELECT id FROM processed_messages WHERE id = '{message_id}' FOR UPDATE;
        UPDATE processed_messages SET processing_attempts = 1 WHERE id = '{message_id}';
        DELETE FROM outbound_messages WHERE id = '{outbound_id}';
        ROLLBACK;
    """
    psql(container, database, runtime, temp, sql)


def assert_denied(container: str, database: str, role: str, temp: Path, sql: str) -> None:
    result = psql(container, database, role, temp, sql, check=False)
    if result.returncode == 0:
        raise CheckFailed("an operation that must be denied succeeded")


def exercise_denials(
    container: str,
    database: str,
    bootstrap: str,
    migrator: str,
    runtime: str,
    backup: str,
    temp: Path,
    suffix: str,
) -> None:
    target = f"deny_target_{suffix}"
    psql(container, database, migrator, temp, f"CREATE TABLE {target} (id integer)")
    for statement in (
        f"CREATE TABLE runtime_create_{suffix} (id integer)",
        f"CREATE TEMP TABLE runtime_temp_{suffix} (id integer)",
        f"CREATE SCHEMA runtime_schema_{suffix}",
        f"ALTER TABLE {target} ADD COLUMN denied integer",
        f"DROP TABLE {target}",
        f"TRUNCATE TABLE {target}",
        f"CREATE ROLE runtime_created_{suffix}",
        f"CREATE DATABASE runtime_created_{suffix}",
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"',
        f"SET ROLE {migrator}",
        f"SET ROLE {bootstrap}",
    ):
        assert_denied(container, database, runtime, temp, statement)
    for statement in (
        f"INSERT INTO {target} VALUES (1)",
        f"UPDATE {target} SET id = 2",
        f"DELETE FROM {target}",
        f"CREATE TABLE backup_create_{suffix} (id integer)",
        f"ALTER TABLE {target} ADD COLUMN backup_denied integer",
        f"DROP TABLE {target}",
        f"TRUNCATE TABLE {target}",
        f"SET ROLE {migrator}",
        f"SET ROLE {bootstrap}",
    ):
        assert_denied(container, database, backup, temp, statement)
    psql(container, database, backup, temp, f"SELECT * FROM {target}")
    psql(container, database, migrator, temp, f"DROP TABLE {target}")


def exercise_default_privileges(
    container: str,
    database: str,
    migrator: str,
    runtime: str,
    backup: str,
    temp: Path,
    suffix: str,
) -> None:
    table = f"future_table_{suffix}"
    sequence = f"future_sequence_{suffix}"
    psql(
        container,
        database,
        migrator,
        temp,
        f"CREATE TABLE {table} (id integer); CREATE SEQUENCE {sequence}",
    )
    checks = psql(
        container,
        database,
        migrator,
        temp,
        f"""
        SELECT
          has_table_privilege('{runtime}', '{table}', 'SELECT,INSERT,UPDATE,DELETE')
          AND NOT has_table_privilege('{runtime}', '{table}', 'TRUNCATE,REFERENCES,TRIGGER')
          AND has_table_privilege('{backup}', '{table}', 'SELECT')
          AND NOT has_table_privilege('{backup}', '{table}', 'INSERT,UPDATE,DELETE,TRUNCATE')
          AND has_sequence_privilege('{runtime}', '{sequence}', 'USAGE,SELECT')
          AND NOT has_sequence_privilege('{runtime}', '{sequence}', 'UPDATE')
          AND has_sequence_privilege('{backup}', '{sequence}', 'SELECT')
          AND NOT has_sequence_privilege('{backup}', '{sequence}', 'USAGE,UPDATE')
          AND NOT EXISTS (
            SELECT 1 FROM pg_class c, aclexplode(c.relacl) acl
            WHERE c.relname IN ('{table}', '{sequence}') AND acl.grantee = 0
          );
        """,
        tuples=True,
    ).stdout.strip()
    if checks != "t":
        raise CheckFailed("future-object default privileges are incorrect")
    psql(container, database, migrator, temp, f"DROP TABLE {table}; DROP SEQUENCE {sequence}")


def exercise_dump(container: str, database: str, backup: str, temp: Path, suffix: str) -> None:
    dump_path = f"/tmp/oink-lp-{suffix}.dump"
    try:
        docker(
            "exec",
            "--env",
            f"ROLE_PASSWORD_FILE=/run/oink-secrets/{backup}.secret",
            "--env",
            f"PGUSER={backup}",
            "--env",
            f"PGDATABASE={database}",
            container,
            "sh",
            "/opt/oink/postgres/with-pgpass.sh",
            "pg_dump",
            "--format=custom",
            "--file",
            dump_path,
        )
        docker("exec", container, "pg_restore", "--list", dump_path)
    finally:
        docker("exec", container, "rm", "-f", dump_path, check=False)


def validate(
    container: str,
    database: str,
    bootstrap: str,
    migrator: str,
    runtime: str,
    backup: str,
    temp: Path,
) -> None:
    provision(
        container,
        "validate-privileges.sh",
        database,
        bootstrap,
        migrator,
        runtime,
        backup,
        temp,
    )


def exercise_password_rotation(
    container: str,
    database: str,
    bootstrap: str,
    migrator: str,
    runtime: str,
    backup: str,
    temp: Path,
    rotated_passwords: dict[str, str],
) -> None:
    for role in (runtime, backup, migrator, bootstrap):
        provision(
            container,
            "rotate-password.sh",
            database,
            bootstrap,
            migrator,
            runtime,
            backup,
            temp,
            {
                "ROTATE_ROLE": role,
                "NEW_PASSWORD_FILE": f"/run/oink-secrets/{role}.rotated.secret",
            },
        )
        write_secret(temp / f"{role}.secret", rotated_passwords[role])
        psql(container, database, role, temp, "SELECT 1")


def exercise_existing_upgrade(
    *,
    postgres: str,
    app_image: str,
    network: str,
    database: str,
    bootstrap: str,
    legacy: str,
    migrator: str,
    runtime: str,
    backup: str,
    passwords: dict[str, str],
    temp: Path,
    suffix: str,
) -> None:
    psql(
        postgres,
        database=database.replace("existing", "new"),
        role=bootstrap,
        temp=temp,
        sql=(
            f"CREATE ROLE {legacy} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
            "NOINHERIT NOREPLICATION NOBYPASSRLS PASSWORD NULL"
        ),
    )
    rotate_legacy_password(
        postgres, database.replace("existing", "new"), bootstrap, legacy, passwords[legacy]
    )
    docker(
        "exec",
        "--env",
        f"ROLE_PASSWORD_FILE=/run/oink-secrets/{bootstrap}.secret",
        "--env",
        f"PGUSER={bootstrap}",
        postgres,
        "sh",
        "/opt/oink/postgres/with-pgpass.sh",
        "createdb",
        "--owner",
        legacy,
        database,
    )
    migration_env = temp / "existing-migration.env"
    migration_env.write_text(
        "APP_ENV=production\n"
        f"MIGRATION_DATABASE_URL={connection_url(legacy, passwords[legacy], database)}\n",
        encoding="utf-8",
        newline="\n",
    )
    run_alembic(app_image, network, migration_env, "upgrade", "20260908_0012")
    psql(
        postgres,
        database,
        legacy,
        temp,
        "INSERT INTO users (id, phone_number) VALUES "
        f"('30000000-0000-4000-8000-{suffix.rjust(12, '0')}', 'upgrade-{suffix}')",
    )
    before = database_digest(postgres, database, legacy, temp)
    historical_before = historical_data_digest(postgres, database, legacy, temp)
    schema_before = schema_digest(postgres, database, legacy, temp)
    extra = {
        "LEGACY_OWNER": legacy,
        "ALLOW_EXISTING_DATABASE_ADAPTATION": "I_UNDERSTAND_THIS_DATABASE_WILL_BE_MODIFIED",
    }
    assert_unexpected_owner_is_rejected(
        postgres,
        database,
        bootstrap,
        legacy,
        migrator,
        runtime,
        backup,
        temp,
        suffix,
        extra,
    )
    provision(
        postgres, "adapt-existing.sh", database, bootstrap, migrator, runtime, backup, temp, extra
    )
    after_first = database_digest(postgres, database, migrator, temp)
    schema_after_first = schema_digest(postgres, database, migrator, temp)
    first_inventory = latest_inventory_digest(temp / "inventory", "ownership-after-")
    provision(
        postgres, "adapt-existing.sh", database, bootstrap, migrator, runtime, backup, temp, extra
    )
    after_second = database_digest(postgres, database, migrator, temp)
    schema_after_second = schema_digest(postgres, database, migrator, temp)
    second_inventory = latest_inventory_digest(temp / "inventory", "ownership-after-")
    if before != after_first or after_first != after_second:
        raise CheckFailed("existing-database data, IDs, or counts changed")
    if schema_before != schema_after_first or schema_after_first != schema_after_second:
        raise CheckFailed("existing-database constraints, indexes, columns, or ENUMs changed")
    if first_inventory != second_inventory:
        raise CheckFailed("repeated adaptation changed ownership inventory")
    upgrade_env = temp / "existing-upgrade.env"
    upgrade_env.write_text(
        "APP_ENV=production\n"
        f"MIGRATION_DATABASE_URL={connection_url(migrator, passwords[migrator], database)}\n",
        encoding="utf-8",
        newline="\n",
    )
    run_alembic(app_image, network, upgrade_env, "upgrade", "head")
    run_alembic(app_image, network, upgrade_env, "current", "--check-heads")
    if historical_data_digest(postgres, database, migrator, temp) != historical_before:
        raise CheckFailed("0012-to-head upgrade changed historical data, IDs, or counts")
    wrong_owner_count = psql(
        postgres,
        database,
        migrator,
        temp,
        f"""
        SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public' AND c.relkind IN ('r','p','S','v','m','f','i','I')
          AND pg_get_userbyid(c.relowner) <> '{migrator}'
        """,
        tuples=True,
    ).stdout.strip()
    if wrong_owner_count != "0":
        raise CheckFailed("existing-database ownership was not fully transferred")
    validate(postgres, database, bootstrap, migrator, runtime, backup, temp)


def assert_unexpected_owner_is_rejected(
    postgres: str,
    database: str,
    bootstrap: str,
    legacy: str,
    migrator: str,
    runtime: str,
    backup: str,
    temp: Path,
    suffix: str,
    extra: dict[str, str],
) -> None:
    intruder = f"oink_intruder_{suffix}"
    table = f"unexpected_owner_{suffix}"
    psql(
        postgres,
        database,
        bootstrap,
        temp,
        f"CREATE ROLE {intruder} NOLOGIN; CREATE TABLE {table} (id integer); "
        f"ALTER TABLE {table} OWNER TO {intruder}",
    )
    result = provision(
        postgres,
        "adapt-existing.sh",
        database,
        bootstrap,
        migrator,
        runtime,
        backup,
        temp,
        extra,
        check=False,
    )
    if result.returncode == 0:
        raise CheckFailed("existing adaptation accepted an unexpected owner")
    managed_role_exists = psql(
        postgres,
        database,
        bootstrap,
        temp,
        f"SELECT EXISTS (SELECT FROM pg_roles WHERE rolname = '{migrator}')",
        tuples=True,
    ).stdout.strip()
    if managed_role_exists != "f":
        raise CheckFailed("existing adaptation changed state before ownership preflight")
    psql(
        postgres,
        database,
        bootstrap,
        temp,
        f"ALTER TABLE {table} OWNER TO {legacy}; DROP TABLE {table}; DROP ROLE {intruder}",
    )


def rotate_legacy_password(
    container: str, database: str, bootstrap: str, legacy: str, password: str
) -> None:
    result = docker(
        "exec",
        "--interactive",
        "--env",
        f"ROLE_PASSWORD_FILE=/run/oink-secrets/{bootstrap}.secret",
        "--env",
        f"PGUSER={bootstrap}",
        "--env",
        f"PGDATABASE={database}",
        container,
        "sh",
        "/opt/oink/postgres/with-pgpass.sh",
        "psql",
        "--no-password",
        "--no-psqlrc",
        "--set=ON_ERROR_STOP=1",
        "--command",
        f"\\password {legacy}",
        input_text=f"{password}\n{password}\n",
        check=False,
    )
    if result.returncode != 0:
        raise CheckFailed("synthetic legacy password setup failed")


def database_digest(container: str, database: str, role: str, temp: Path) -> str:
    values = psql(
        container,
        database,
        role,
        temp,
        """
        SELECT table_name || ':' || count_value || ':' || digest_value FROM (
          SELECT 'alembic_version' AS table_name, count(*)::text AS count_value,
                 md5(coalesce(string_agg(version_num, ',' ORDER BY version_num), ''))
                   AS digest_value
          FROM alembic_version
          UNION ALL
          SELECT 'users', count(*)::text,
                 md5(coalesce(string_agg(id::text || ':' || phone_number, ',' ORDER BY id), ''))
          FROM users
          UNION ALL
          SELECT 'categories', count(*)::text,
                 md5(coalesce(string_agg(id::text || ':' || name, ',' ORDER BY id), ''))
          FROM categories
          UNION ALL
          SELECT 'all_table_counts', '1', md5(concat_ws(',',
                 (SELECT count(*) FROM alembic_version),
                 (SELECT count(*) FROM categories),
                 (SELECT count(*) FROM conversation_states),
                 (SELECT count(*) FROM expense_history),
                 (SELECT count(*) FROM expenses),
                 (SELECT count(*) FROM outbound_messages),
                 (SELECT count(*) FROM processed_messages),
                 (SELECT count(*) FROM users)))
        ) data ORDER BY table_name
        """,
        tuples=True,
    ).stdout
    return hashlib.sha256(values.encode()).hexdigest()


def historical_data_digest(container: str, database: str, role: str, temp: Path) -> str:
    values = psql(
        container,
        database,
        role,
        temp,
        """
        SELECT concat_ws(',',
          (SELECT count(*)::text FROM categories),
          (SELECT count(*)::text FROM conversation_states),
          (SELECT count(*)::text FROM expense_history),
          (SELECT count(*)::text FROM expenses),
          (SELECT count(*)::text FROM outbound_messages),
          (SELECT count(*)::text FROM processed_messages),
          (SELECT count(*)::text FROM users),
          (SELECT coalesce(string_agg(id::text || ':' || phone_number, ',' ORDER BY id), '')
             FROM users))
        """,
        tuples=True,
    ).stdout
    return hashlib.sha256(values.encode()).hexdigest()


def schema_digest(container: str, database: str, role: str, temp: Path) -> str:
    values = psql(
        container,
        database,
        role,
        temp,
        """
        SELECT kind || ':' || name || ':' || definition FROM (
          SELECT 'column' AS kind, c.table_name || '.' || c.column_name AS name,
                 c.data_type || ':' || c.is_nullable || ':' || coalesce(c.column_default, '')
                   AS definition
          FROM information_schema.columns c WHERE c.table_schema = 'public'
          UNION ALL
          SELECT 'constraint', con.conname, pg_get_constraintdef(con.oid)
          FROM pg_constraint con JOIN pg_namespace n ON n.oid = con.connamespace
          WHERE n.nspname = 'public'
          UNION ALL
          SELECT 'index', indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'
          UNION ALL
          SELECT 'enum', t.typname || '.' || e.enumsortorder::text, e.enumlabel
          FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
          JOIN pg_enum e ON e.enumtypid = t.oid WHERE n.nspname = 'public'
        ) definitions ORDER BY kind, name, definition
        """,
        tuples=True,
    ).stdout
    return hashlib.sha256(values.encode()).hexdigest()


def latest_inventory_digest(directory: Path, prefix: str) -> str:
    paths = sorted(directory.glob(f"{prefix}*"), key=lambda path: path.stat().st_mtime_ns)
    if not paths:
        raise CheckFailed("ownership inventory was not produced")
    return hashlib.sha256(paths[-1].read_bytes()).hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
