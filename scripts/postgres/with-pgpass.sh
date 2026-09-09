#!/bin/sh
set -eu
set +x
umask 077

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIRECTORY/_common.sh"

[ -n "${PGUSER-}" ] || { printf '%s\n' "ERROR: PGUSER is required" >&2; exit 1; }
[ -n "${ROLE_PASSWORD_FILE-}" ] || {
    printf '%s\n' "ERROR: ROLE_PASSWORD_FILE is required" >&2
    exit 1
}
[ -f "$ROLE_PASSWORD_FILE" ] || {
    printf '%s\n' "ERROR: role password file is missing" >&2
    exit 1
}

require_identifier PGUSER
read_secret_file "$ROLE_PASSWORD_FILE"
role_password=$SECRET_VALUE
unset SECRET_VALUE normalized_secret
escaped_password=$(printf '%s' "$role_password" | sed 's/\\/\\\\/g; s/:/\\:/g')
unset role_password

PGPASSFILE=$(mktemp)
export PGPASSFILE
trap 'rm -f -- "$PGPASSFILE"' EXIT HUP INT TERM
printf '*:*:*:%s:%s\n' "$PGUSER" "$escaped_password" > "$PGPASSFILE"
unset escaped_password PGPASSWORD
export PGHOST=${PGHOST:-localhost}

"$@"
