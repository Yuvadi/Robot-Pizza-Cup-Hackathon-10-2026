#!/usr/bin/env bash
# Guide manual calibration and save verified leader/follower profiles.
# Hardware calibration actions are performed explicitly in the Station UI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=load_config.sh
source "$ROOT/scripts/load_config.sh"
require_var FOLLOWER_SERIAL
require_var LEADER_SERIAL

exec python3 "$ROOT/calibration.py" \
  --server "$STATION_HOST" --norma-core "$NORMA_CORE_ROOT" \
  --leader-serial "$LEADER_SERIAL" --follower-serial "$FOLLOWER_SERIAL" "$@"
