#!/bin/sh
set -eu
set +x
umask 077

die() {
    printf '%s\n' "ERROR: $1" >&2
    exit 1
}

require_value() {
    variable_name=$1
    eval "variable_value=\${$variable_name-}"
    [ -n "$variable_value" ] || die "$variable_name is required"
}

require_identifier() {
    variable_name=$1
    eval "identifier=\${$variable_name-}"
    require_value "$variable_name"
    case "$identifier" in
        [A-Za-z_]* ) ;;
        * ) die "$variable_name must be a safe PostgreSQL identifier" ;;
    esac
    case "$identifier" in
        *[!A-Za-z0-9_]* ) die "$variable_name must be a safe PostgreSQL identifier" ;;
    esac
    [ "${#identifier}" -le 63 ] || die "$variable_name exceeds PostgreSQL's identifier limit"
}

require_role_model() {
    for variable_name in \
        APP_DATABASE APP_SCHEMA BOOTSTRAP_USER MIGRATOR_ROLE RUNTIME_ROLE BACKUP_ROLE
    do
        require_identifier "$variable_name"
    done

    for role_name in "$MIGRATOR_ROLE" "$RUNTIME_ROLE" "$BACKUP_ROLE"; do
        case "$role_name" in
            pg_* | postgres | public ) die "application roles use a reserved identifier" ;;
        esac
    done
    case "$APP_DATABASE" in
        postgres | template0 | template1 ) die "APP_DATABASE uses a reserved identifier" ;;
    esac

    [ "$MIGRATOR_ROLE" != "$RUNTIME_ROLE" ] || die "roles must be distinct"
    [ "$MIGRATOR_ROLE" != "$BACKUP_ROLE" ] || die "roles must be distinct"
    [ "$RUNTIME_ROLE" != "$BACKUP_ROLE" ] || die "roles must be distinct"
    [ "$BOOTSTRAP_USER" != "$MIGRATOR_ROLE" ] || die "bootstrap must be a separate role"
    [ "$BOOTSTRAP_USER" != "$RUNTIME_ROLE" ] || die "bootstrap must be a separate role"
    [ "$BOOTSTRAP_USER" != "$BACKUP_ROLE" ] || die "bootstrap must be a separate role"

    require_value BOOTSTRAP_HOST
    require_value BOOTSTRAP_PORT
    case "$BOOTSTRAP_PORT" in
        *[!0-9]* | "" ) die "BOOTSTRAP_PORT must be numeric" ;;
    esac
}

read_secret_file() {
    secret_path=$1
    [ -f "$secret_path" ] || die "a required password file is missing"
    [ ! -L "$secret_path" ] || die "password files must not be symbolic links"
    SECRET_VALUE=
    {
        IFS= read -r SECRET_VALUE || [ -n "$SECRET_VALUE" ]
        if IFS= read -r _extra_line; then
            die "password files must contain exactly one line"
        fi
    } < "$secret_path"
    [ -n "$SECRET_VALUE" ] || die "a password file is empty"
    [ "${#SECRET_VALUE}" -ge 20 ] || die "a password does not meet the minimum length"
    normalized_secret=$(printf '%s' "$SECRET_VALUE" | tr '[:upper:]' '[:lower:]')
    case "$normalized_secret" in
        *change-me* | *changeme* | *replace-me* | *placeholder* | your-* )
            die "a password file contains a placeholder"
            ;;
    esac
}

configure_admin_connection() {
    require_value BOOTSTRAP_PASSWORD_FILE
    read_secret_file "$BOOTSTRAP_PASSWORD_FILE"
    escaped_password=$(printf '%s' "$SECRET_VALUE" | sed 's/\\/\\\\/g; s/:/\\:/g')
    unset SECRET_VALUE normalized_secret

    PGPASSFILE=$(mktemp)
    export PGPASSFILE
    printf '*:*:*:%s:%s\n' "$BOOTSTRAP_USER" "$escaped_password" > "$PGPASSFILE"
    unset escaped_password
    export PGHOST=$BOOTSTRAP_HOST
    export PGPORT=$BOOTSTRAP_PORT
    export PGUSER=$BOOTSTRAP_USER
    export PGDATABASE=$APP_DATABASE
    export PGSSLMODE=${BOOTSTRAP_SSLMODE:-prefer}
    unset PGPASSWORD
}

cleanup_common() {
    if [ -n "${PGPASSFILE-}" ] && [ -f "$PGPASSFILE" ]; then
        rm -f -- "$PGPASSFILE"
    fi
}

admin_psql() {
    psql --no-password --no-psqlrc --set=ON_ERROR_STOP=1 --quiet "$@"
}

admin_query() {
    query=$1
    shift
    printf '%s\n' "$query" | admin_psql "$@"
}

role_exists() {
    admin_query "SELECT EXISTS (SELECT FROM pg_roles WHERE rolname = :'role_name')" \
        --tuples-only --no-align --set=role_name="$1"
}

role_has_password() {
    admin_query \
        "SELECT rolpassword IS NOT NULL FROM pg_authid WHERE rolname = :'role_name'" \
        --tuples-only --no-align --set=role_name="$1"
}

assert_existing_role_is_managed() {
    role_name=$1
    expected_comment=$2
    existing=$3
    [ "$existing" != "t" ] && return
    actual_comment=$(admin_query \
        "SELECT coalesce(shobj_description(oid, 'pg_authid'), '') FROM pg_roles WHERE rolname = :'role_name'" \
        --tuples-only --no-align --set=role_name="$role_name")
    [ "$actual_comment" = "$expected_comment" ] || \
        die "an existing application role is not marked as managed by Oink FinAI"
}

assert_roles_have_no_memberships() {
    membership_count=$(admin_query \
        "SELECT count(*) FROM pg_auth_members m JOIN pg_roles r ON r.oid IN (m.member, m.roleid) WHERE r.rolname IN (:'migrator_role', :'runtime_role', :'backup_role')" \
        --tuples-only --no-align \
        --set=migrator_role="$MIGRATOR_ROLE" \
        --set=runtime_role="$RUNTIME_ROLE" \
        --set=backup_role="$BACKUP_ROLE")
    [ "$membership_count" = "0" ] || die "an application role has an unexpected membership"
}

assert_cluster_baseline() {
    trust_rule_count=$(admin_query \
        "SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL OR auth_method = 'trust'" \
        --tuples-only --no-align)
    [ "$trust_rule_count" = "0" ] || die "pg_hba.conf contains invalid or trust authentication rules"

    public_database_count=$(admin_query "
        SELECT count(*)
        FROM pg_database d
        WHERE d.datallowconn
          AND d.datname <> current_database()
          AND EXISTS (
              SELECT 1
              FROM aclexplode(coalesce(d.datacl, acldefault('d', d.datdba))) acl
              WHERE acl.grantee = 0 AND acl.privilege_type = 'CONNECT'
          )" --tuples-only --no-align)
    [ "$public_database_count" = "0" ] || \
        die "PUBLIC can connect to another database; dedicated-cluster baseline is required"
}

rotate_role_password() {
    target_role=$1
    password_file=$2
    read_secret_file "$password_file"
    password_value=$SECRET_VALUE
    unset SECRET_VALUE normalized_secret
    error_file=$(mktemp)
    if ! {
        printf '%s\n%s\n' "$password_value" "$password_value"
    } | admin_psql --command=BEGIN --command="\\password $target_role" --command=COMMIT \
        > /dev/null 2> "$error_file"; then
        unset password_value
        rm -f -- "$error_file"
        die "password rotation failed; PostgreSQL details were suppressed"
    fi
    unset password_value
    rm -f -- "$error_file"
}

ensure_roles() {
    migrator_existed=$(role_exists "$MIGRATOR_ROLE")
    runtime_existed=$(role_exists "$RUNTIME_ROLE")
    backup_existed=$(role_exists "$BACKUP_ROLE")
    assert_existing_role_is_managed "$MIGRATOR_ROLE" "oink-finai:migrator:v1" "$migrator_existed"
    assert_existing_role_is_managed "$RUNTIME_ROLE" "oink-finai:runtime:v1" "$runtime_existed"
    assert_existing_role_is_managed "$BACKUP_ROLE" "oink-finai:backup:v1" "$backup_existed"
    assert_roles_have_no_memberships

    admin_psql \
        --set=migrator_role="$MIGRATOR_ROLE" \
        --set=runtime_role="$RUNTIME_ROLE" \
        --set=backup_role="$BACKUP_ROLE" \
        --file="$(dirname "$0")/roles.sql"

    [ "$(role_has_password "$MIGRATOR_ROLE")" = "t" ] || \
        rotate_role_password "$MIGRATOR_ROLE" "$MIGRATOR_PASSWORD_FILE"
    [ "$(role_has_password "$RUNTIME_ROLE")" = "t" ] || \
        rotate_role_password "$RUNTIME_ROLE" "$RUNTIME_PASSWORD_FILE"
    [ "$(role_has_password "$BACKUP_ROLE")" = "t" ] || \
        rotate_role_password "$BACKUP_ROLE" "$BACKUP_PASSWORD_FILE"
}

configure_database_privileges() {
    admin_psql --single-transaction \
        --set=app_database="$APP_DATABASE" \
        --set=app_schema="$APP_SCHEMA" \
        --set=migrator_role="$MIGRATOR_ROLE" \
        --set=runtime_role="$RUNTIME_ROLE" \
        --set=backup_role="$BACKUP_ROLE" \
        --file="$(dirname "$0")/privileges.sql"
}
