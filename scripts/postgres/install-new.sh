#!/bin/sh
set -eu
set +x

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIRECTORY/_common.sh"

require_role_model
require_value MIGRATOR_PASSWORD_FILE
require_value RUNTIME_PASSWORD_FILE
require_value BACKUP_PASSWORD_FILE
configure_admin_connection
trap cleanup_common EXIT HUP INT TERM
assert_cluster_baseline

object_count=$(admin_query "
        SELECT count(*)
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_roles r ON r.oid = c.relowner
        WHERE n.nspname = :'app_schema'
          AND c.relkind IN ('r', 'p', 'S', 'v', 'm', 'f')
          AND r.rolname <> :'migrator_role';" \
    --tuples-only --no-align --set=app_schema="$APP_SCHEMA" \
    --set=migrator_role="$MIGRATOR_ROLE")
[ "$object_count" = "0" ] || die "new-install preflight found objects not owned by migrator"

ensure_roles

schema_exists=$(admin_query \
    "SELECT EXISTS (SELECT FROM pg_namespace WHERE nspname = :'app_schema')" \
    --tuples-only --no-align --set=app_schema="$APP_SCHEMA")
if [ "$schema_exists" != "t" ]; then
    admin_query 'BEGIN; CREATE SCHEMA :"app_schema" AUTHORIZATION :"migrator_role"; COMMIT;' \
        --set=app_schema="$APP_SCHEMA" --set=migrator_role="$MIGRATOR_ROLE"
fi

configure_database_privileges
printf '%s\n' "New database privilege model installed."
