#!/bin/sh
set -eu
set +x

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIRECTORY/_common.sh"

require_role_model
require_value ROTATE_ROLE
require_value NEW_PASSWORD_FILE
case "$ROTATE_ROLE" in
    "$BOOTSTRAP_USER" | "$MIGRATOR_ROLE" | "$RUNTIME_ROLE" | "$BACKUP_ROLE" ) ;;
    * ) die "ROTATE_ROLE must be one of the four managed roles" ;;
esac

configure_admin_connection
trap cleanup_common EXIT HUP INT TERM
[ "$(role_exists "$ROTATE_ROLE")" = "t" ] || die "target role does not exist"
rotate_role_password "$ROTATE_ROLE" "$NEW_PASSWORD_FILE"
printf '%s\n' "Password rotated for the requested application role."
