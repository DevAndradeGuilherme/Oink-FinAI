#!/bin/sh
set -eu
set +x

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIRECTORY/_common.sh"

require_role_model
configure_admin_connection
trap cleanup_common EXIT HUP INT TERM
assert_cluster_baseline

admin_psql --set=app_schema="$APP_SCHEMA" \
    --set=migrator_role="$MIGRATOR_ROLE" \
    --set=runtime_role="$RUNTIME_ROLE" \
    --set=backup_role="$BACKUP_ROLE" \
    --file="$SCRIPT_DIRECTORY/validate-privileges.sql"

failure_count=$(admin_query "
        SELECT
            3 - (SELECT count(*) FROM pg_roles
                  WHERE rolname IN (:'migrator_role', :'runtime_role', :'backup_role'))
          + (SELECT count(*) FROM pg_roles
              WHERE rolname IN (:'migrator_role', :'runtime_role', :'backup_role')
                AND (NOT rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole
                     OR rolreplication OR rolbypassrls))
          + (SELECT count(*) FROM pg_auth_members m JOIN pg_roles r ON r.oid IN (m.member, m.roleid)
              WHERE r.rolname IN (:'migrator_role', :'runtime_role', :'backup_role'))
          + (SELECT count(*) FROM pg_database d
              CROSS JOIN (VALUES (:'migrator_role'), (:'runtime_role'), (:'backup_role')) roles(role_name)
             WHERE d.datallowconn AND d.datname <> current_database()
               AND has_database_privilege(roles.role_name, d.datname, 'CONNECT'))
          + CASE WHEN has_schema_privilege(:'runtime_role', :'app_schema', 'CREATE') THEN 1 ELSE 0 END
          + CASE WHEN has_schema_privilege(:'backup_role', :'app_schema', 'CREATE') THEN 1 ELSE 0 END;" \
    --tuples-only --no-align --set=app_schema="$APP_SCHEMA" \
    --set=migrator_role="$MIGRATOR_ROLE" \
    --set=runtime_role="$RUNTIME_ROLE" \
    --set=backup_role="$BACKUP_ROLE")
[ "$failure_count" = "0" ] || die "least-privilege validation failed"
