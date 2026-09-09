#!/bin/sh
set -eu
set +x

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
case "${PROVISION_MODE-}" in
    new ) exec sh "$SCRIPT_DIRECTORY/install-new.sh" ;;
    existing ) exec sh "$SCRIPT_DIRECTORY/adapt-existing.sh" ;;
    * ) printf '%s\n' "ERROR: PROVISION_MODE must be new or existing" >&2; exit 1 ;;
esac
