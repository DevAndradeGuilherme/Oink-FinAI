ALTER DATABASE :"app_database" OWNER TO :"migrator_role";
REVOKE ALL ON DATABASE :"app_database" FROM PUBLIC;
REVOKE ALL ON DATABASE :"app_database" FROM :"runtime_role", :"backup_role";
GRANT CONNECT ON DATABASE :"app_database" TO :"runtime_role", :"backup_role";

ALTER SCHEMA :"app_schema" OWNER TO :"migrator_role";
REVOKE ALL ON SCHEMA :"app_schema" FROM PUBLIC;
REVOKE ALL ON SCHEMA :"app_schema" FROM :"runtime_role", :"backup_role";
GRANT USAGE ON SCHEMA :"app_schema" TO :"runtime_role", :"backup_role";

REVOKE ALL ON ALL TABLES IN SCHEMA :"app_schema" FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA :"app_schema" FROM :"runtime_role", :"backup_role";
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA :"app_schema"
    TO :"runtime_role";
GRANT SELECT ON ALL TABLES IN SCHEMA :"app_schema" TO :"backup_role";

SELECT format(
    'REVOKE ALL PRIVILEGES ON TABLE %I.alembic_version FROM %I',
    :'app_schema',
    :'runtime_role'
)
WHERE to_regclass(format('%I.alembic_version', :'app_schema')) IS NOT NULL
\gexec

REVOKE ALL ON ALL SEQUENCES IN SCHEMA :"app_schema" FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA :"app_schema" FROM :"runtime_role", :"backup_role";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA :"app_schema" TO :"runtime_role";
GRANT SELECT ON ALL SEQUENCES IN SCHEMA :"app_schema" TO :"backup_role";

SELECT format(
    'REVOKE ALL ON TYPE %I.%I FROM PUBLIC, %I, %I',
    n.nspname, t.typname, :'runtime_role', :'backup_role'
)
FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = :'app_schema' AND t.typrelid = 0 AND t.typtype IN ('c','d','e','m','r')
\gexec
SELECT format(
    'GRANT USAGE ON TYPE %I.%I TO %I, %I',
    n.nspname, t.typname, :'runtime_role', :'backup_role'
)
FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = :'app_schema' AND t.typrelid = 0 AND t.typtype IN ('c','d','e','m','r')
\gexec

REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA :"app_schema" FROM PUBLIC;

ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"runtime_role";
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    GRANT SELECT ON TABLES TO :"backup_role";

ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    GRANT USAGE, SELECT ON SEQUENCES TO :"runtime_role";
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    GRANT SELECT ON SEQUENCES TO :"backup_role";

ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    REVOKE ALL ON TYPES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    GRANT USAGE ON TYPES TO :"runtime_role", :"backup_role";

ALTER DEFAULT PRIVILEGES FOR ROLE :"migrator_role" IN SCHEMA :"app_schema"
    REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;

ALTER ROLE :"migrator_role" IN DATABASE :"app_database"
    SET search_path TO :"app_schema", pg_catalog;
ALTER ROLE :"runtime_role" IN DATABASE :"app_database"
    SET search_path TO :"app_schema", pg_catalog;
ALTER ROLE :"backup_role" IN DATABASE :"app_database"
    SET search_path TO :"app_schema", pg_catalog;
ALTER ROLE :"backup_role" IN DATABASE :"app_database"
    SET default_transaction_read_only TO on;
