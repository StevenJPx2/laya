#!/usr/bin/env bash
set -euo pipefail

readonly LABEL="com.laya.daemon"
readonly SOCKET="$HOME/Library/Application Support/laya/laya.sock"
readonly INSTALL_DIR="$HOME/Library/Application Support/laya"
readonly PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_RELEASE="$SOURCE_ROOT/.build/release"
SKIP_BUILD=0
WAIT_SECONDS="${LAYA_WAIT_SECONDS:-600}"
MODEL_SOURCE=""
MODEL_NAME=""

die() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: tools/install-daemon.sh [--no-build | --status | --uninstall]

Install the compiled Laya daemon as a per-user macOS launchd service.

  --no-build   Reuse .build/release/laya-daemon.
  --status     Show launchd and socket health without changing anything.
  --uninstall  Stop the service and remove its installed binary/model.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

service_target() {
  printf 'gui/%s/%s' "$(id -u)" "$LABEL"
}

print_status() {
  if launchctl print "$(service_target)" 2>/dev/null | sed -n '1,18p'; then
    :
  else
    printf 'service: not loaded\n'
  fi

  if [[ -S "$SOCKET" ]]; then
    local health
    health="$(printf '{"op":"health"}\n' | nc -U "$SOCKET" 2>/dev/null || true)"
    printf 'socket: %s\n' "${health:-unavailable}"
  else
    printf 'socket: not ready (%s)\n' "$SOCKET"
  fi
}

write_plist() {
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$INSTALL_DIR/laya-daemon</string>
    <string>$INSTALL_DIR/$MODEL_NAME</string>
    <string>$INSTALL_DIR/assets</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/laya-daemon.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/laya-daemon.error.log</string>
</dict></plist>
PLIST

  plutil -lint "$PLIST" >/dev/null
}

build_release() {
  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    printf 'Building release daemon...\n'
    swift build -c release --package-path "$SOURCE_ROOT"
  fi

  [[ -x "$BUILD_RELEASE/laya-daemon" ]] || die "release binary not found; run swift build -c release"
  [[ -d "$SOURCE_ROOT/build/assets" ]] || die "build/assets not found; export the model first"

  if [[ -d "$SOURCE_ROOT/build/laya.mlmodelc" && "$SOURCE_ROOT/build/laya.mlmodelc" -nt "$SOURCE_ROOT/build/laya.mlpackage" ]]; then
    MODEL_SOURCE="$SOURCE_ROOT/build/laya.mlmodelc"
    MODEL_NAME="laya.mlmodelc"
  else
    [[ -d "$SOURCE_ROOT/build/laya.mlpackage" ]] || die "build/laya.mlpackage not found; export the model first"
    MODEL_SOURCE="$SOURCE_ROOT/build/laya.mlpackage"
    MODEL_NAME="laya.mlpackage"
  fi
}

install_payload() {
  local stage
  stage="$(mktemp -d "${TMPDIR:-/tmp}/laya-install.XXXXXX")"

  printf 'Staging daemon and model...\n'
  cp "$BUILD_RELEASE/laya-daemon" "$stage/laya-daemon"
  chmod 755 "$stage/laya-daemon"
  ditto "$MODEL_SOURCE" "$stage/$MODEL_NAME"
  ditto "$SOURCE_ROOT/build/assets" "$stage/assets"

  mkdir -p "$INSTALL_DIR"
  rm -f "$INSTALL_DIR/laya-daemon"
  rm -rf "$INSTALL_DIR/laya.mlpackage" "$INSTALL_DIR/laya.mlmodelc" "$INSTALL_DIR/assets"
  mv "$stage/laya-daemon" "$INSTALL_DIR/laya-daemon"
  mv "$stage/$MODEL_NAME" "$INSTALL_DIR/$MODEL_NAME"
  mv "$stage/assets" "$INSTALL_DIR/assets"
  rm -rf "$stage"
}

start_service() {
  local attempt

  for attempt in 1 2 3; do
    launchctl bootout "$(service_target)" 2>/dev/null || true
    rm -f "$SOCKET"
    if launchctl bootstrap "gui/$(id -u)" "$PLIST"; then
      launchctl kickstart -k "$(service_target)"
      return
    fi
    sleep 1
  done

  die "launchd could not bootstrap $LABEL"
}

wait_for_health() {
  local health

  printf 'Waiting for Core ML model warm-up (up to %ss)' "$WAIT_SECONDS"
  for _ in $(seq 1 "$WAIT_SECONDS"); do
    if [[ -S "$SOCKET" ]]; then
      health="$(printf '{"op":"health"}\n' | nc -U "$SOCKET" 2>/dev/null || true)"
      if [[ "$health" == *'"status":"ok"'* ]]; then
        printf '\n%s\n' "$health"
        return
      fi
    fi
    printf '.'
    sleep 1
  done

  printf '\n'
  tail -40 "$HOME/Library/Logs/laya-daemon.error.log" 2>/dev/null || true
  die "daemon did not become healthy within $WAIT_SECONDS seconds"
}

uninstall() {
  launchctl bootout "$(service_target)" 2>/dev/null || true
  rm -f "$PLIST"
  rm -rf "$INSTALL_DIR/laya-daemon" "$INSTALL_DIR/laya.mlpackage" "$INSTALL_DIR/laya.mlmodelc" "$INSTALL_DIR/assets"
  rm -f "$SOCKET"
  printf 'Removed %s and its installed payload.\n' "$LABEL"
}

main() {
  local action="install"

  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --no-build) SKIP_BUILD=1 ;;
      --status) action="status" ;;
      --uninstall) action="uninstall" ;;
      --help|-h) usage; return 0 ;;
      *) usage >&2; die "unknown option: $1" ;;
    esac
    shift
  done

  if [[ "$action" == "status" ]]; then
    print_status
    return 0
  fi

  [[ "$(uname -s)" == "Darwin" ]] || die "the daemon installer requires macOS"
  require_command launchctl
  require_command plutil
  require_command nc

  if [[ "$action" == "uninstall" ]]; then
    uninstall
    return 0
  fi

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    require_command swift
  fi
  require_command ditto
  build_release
  install_payload
  write_plist
  start_service
  wait_for_health
  printf 'Installed launchd service: %s\n' "$LABEL"
  printf 'Socket: %s\n' "$SOCKET"
}

main "$@"
