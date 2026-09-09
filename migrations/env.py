import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from oink_finai.config.migration import migration_database_url
from oink_finai.database import models  # noqa: F401
from oink_finai.database.base import Base
from oink_finai.observability import configure_application_logging

config = context.config
config.set_main_option("sqlalchemy.url", migration_database_url().replace("%", "%%"))

if os.environ.get("APP_ENV", "development").strip().casefold() == "production":
    configure_application_logging(
        log_format="json",
        level=os.environ.get("LOG_LEVEL", "INFO"),
        service="migrate",
        include_traceback=False,
    )
elif config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()
        if connection.dialect.name == "postgresql":
            _harden_alembic_metadata(connection)


def _harden_alembic_metadata(connection) -> None:
    """Expose only the schema version needed for readiness to tagged runtime roles."""
    schema = connection.exec_driver_sql("SELECT current_schema()").scalar_one()
    runtime_roles = connection.exec_driver_sql(
        """
        SELECT rolname
        FROM pg_roles
        WHERE shobj_description(oid, 'pg_authid') = 'oink-finai:runtime:v1'
        """
    ).scalars()
    quote = connection.dialect.identifier_preparer.quote
    table = f"{quote(schema)}.{quote('alembic_version')}"
    for role in runtime_roles:
        connection.exec_driver_sql(f"REVOKE ALL PRIVILEGES ON TABLE {table} FROM {quote(role)}")
        connection.exec_driver_sql(f"GRANT SELECT ON TABLE {table} TO {quote(role)}")


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
