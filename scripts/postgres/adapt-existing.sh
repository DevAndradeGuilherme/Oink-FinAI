#!/bin/sh
set -eu
set +x

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIRECTORY/_common.sh"

require_role_model
require_identifier LEGACY_OWNER
require_value MIGRATOR_PASSWORD_FILE
require_value RUNTIME_PASSWORD_FILE
require_value BACKUP_PASSWORD_FILE
require_value INVENTORY_DIRECTORY
[ "${ALLOW_EXISTING_DATABASE_ADAPTATION-}" = "I_UNDERSTAND_THIS_DATABASE_WILL_BE_MODIFIED" ] || \
    die "explicit existing-database adaptation marker is required"
[ -d "$INVENTORY_DIRECTORY" ] || die "INVENTORY_DIRECTORY must already exist"

configure_admin_connection
trap cleanup_common EXIT HUP INT TERM
assert_cluster_baseline

schema_exists=$(admin_query \
    "SELECT EXISTS (SELECT FROM pg_namespace WHERE nspname = :'app_schema')" \
    --tuples-only --no-align --set=app_schema="$APP_SCHEMA")
[ "$schema_exists" = "t" ] || die "application schema does not exist"

unexpected_count=$(admin_query "
        WITH owners(owner) AS (
            SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = current_database()
            UNION ALL
            SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = :'app_schema'
            UNION ALL
            SELECT pg_get_userbyid(c.relowner) FROM pg_class c
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = :'app_schema' AND c.relkind IN ('r','p','S','v','m','f','i','I')
            UNION ALL
            SELECT pg_get_userbyid(t.typowner) FROM pg_type t
              JOIN pg_namespace n ON n.oid = t.typnamespace
             WHERE n.nspname = :'app_schema' AND t.typrelid = 0
               AND t.typtype IN ('c','d','e','m','r')
            UNION ALL
            SELECT pg_get_userbyid(p.proowner) FROM pg_proc p
              JOIN pg_namespace n ON n.oid = p.pronamespace
             WHERE n.nspname = :'app_schema'
        )
        SELECT count(*) FROM owners
         WHERE owner NOT IN (:'legacy_owner', :'migrator_role', 'pg_database_owner');" \
    --tuples-only --no-align \
    --set=app_schema="$APP_SCHEMA" --set=legacy_owner="$LEGACY_OWNER" \
    --set=migrator_role="$MIGRATOR_ROLE")
[ "$unexpected_count" = "0" ] || die "unexpected ownership found; no changes were made"

before_inventory=$(mktemp "$INVENTORY_DIRECTORY/ownership-before-XXXXXXXX")
after_inventory=$(mktemp "$INVENTORY_DIRECTORY/ownership-after-XXXXXXXX")
admin_psql --csv --set=app_schema="$APP_SCHEMA" --file="$SCRIPT_DIRECTORY/inventory.sql" \
    > "$before_inventory"

ensure_roles
admin_psql --single-transaction \
    --set=app_database="$APP_DATABASE" --set=app_schema="$APP_SCHEMA" \
    --set=migrator_role="$MIGRATOR_ROLE" --set=runtime_role="$RUNTIME_ROLE" \
    --set=backup_role="$BACKUP_ROLE" \
    --file="$SCRIPT_DIRECTORY/transfer-ownership.sql" \
    --file="$SCRIPT_DIRECTORY/privileges.sql"

admin_psql --csv --set=app_schema="$APP_SCHEMA" --file="$SCRIPT_DIRECTORY/inventory.sql" \
    > "$after_inventory"
printf '%s\n' "Existing database adapted; ownership inventories were written to the requested directory."
