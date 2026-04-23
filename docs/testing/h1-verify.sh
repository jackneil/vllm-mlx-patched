#!/usr/bin/env bash
# H1 verification script — runs N staggered pairs against a model
# served at $MODEL_URL and reports HANG rate.
#
# Usage:
#   NEURAL_ARENA_KEY=<key> \
#     MODEL="mlx-community/Qwen3.6-35B-A3B-4bit" \
#     PAIRS=30 \
#     BG_LOAD=0 \
#     MODEL_URL=https://llm.hank.ai/v1/messages \
#     ./docs/testing/h1-verify.sh
#
# Success criterion per run: 0/<PAIRS> HANG.  For a robust closure
# claim on an intermittent bug, run three independent invocations
# (30 × 3 = 90 staggered pairs total) AND a separate invocation with
# BG_LOAD=5 for the background-load variant.  Any HANG = FAIL.
#
# Requires: jq, curl, python3.  Fixtures at
# /tmp/cc-cap-0[12]-hipaa.json (regenerate via hank-secure-llm
# capture proxy if missing).
set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
PAIRS="${PAIRS:-30}"
BG_LOAD="${BG_LOAD:-0}"
KEY="${NEURAL_ARENA_KEY:?NEURAL_ARENA_KEY not set}"
URL="${MODEL_URL:-https://llm.hank.ai/v1/messages}"
OUT=$(mktemp -d)

LIGHT_FIXTURE="${LIGHT_FIXTURE:-/tmp/cc-cap-01-hipaa.json}"
HEAVY_FIXTURE="${HEAVY_FIXTURE:-/tmp/cc-cap-02-hipaa.json}"
[ -f "$LIGHT_FIXTURE" ] && [ -f "$HEAVY_FIXTURE" ] || {
  echo "missing fixtures: $LIGHT_FIXTURE and/or $HEAVY_FIXTURE"
  echo "regenerate via hank-secure-llm capture proxy"
  exit 2
}

LIGHT=$(mktemp)
HEAVY=$(mktemp)
trap 'rm -rf "$OUT" "$LIGHT" "$HEAVY"; [ -f /tmp/.h1-bg-flag-$$ ] && rm /tmp/.h1-bg-flag-$$' EXIT

jq --arg m "$MODEL" '.model = $m' "$LIGHT_FIXTURE" > "$LIGHT"
jq --arg m "$MODEL" '.model = $m' "$HEAVY_FIXTURE" > "$HEAVY"

if [ "$BG_LOAD" -gt 0 ]; then
  touch "/tmp/.h1-bg-flag-$$"
  for _ in $(seq 1 "$BG_LOAD"); do
    (
      while [ -f "/tmp/.h1-bg-flag-$$" ]; do
        curl -sS -m 20 -N -o /dev/null \
          -X POST "$URL" \
          -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
          -d @"$LIGHT" >/dev/null 2>&1 || true
        sleep 0.1
      done
    ) &
  done
fi

HANGS=0
for i in $(seq 1 "$PAIRS"); do
  L="$OUT/p$i-l.sse"
  H="$OUT/p$i-h.sse"
  ( curl -sS -m 35 -N -o "$L" -X POST "$URL" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d @"$LIGHT" ) &
  sleep 0.05
  ( curl -sS -m 35 -N -o "$H" -X POST "$URL" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d @"$HEAVY" ) &
  wait
  LS=$(grep -c message_stop "$L" 2>/dev/null || echo 0)
  HS=$(grep -c message_stop "$H" 2>/dev/null || echo 0)
  if [ "$LS" -eq 0 ] || [ "$HS" -eq 0 ]; then
    HANGS=$((HANGS+1))
    echo "  pair=$i HANG (light_stops=$LS heavy_stops=$HS)"
  fi
done

[ -f "/tmp/.h1-bg-flag-$$" ] && rm "/tmp/.h1-bg-flag-$$"
wait 2>/dev/null || true

echo
echo "RESULT: $HANGS / $PAIRS HANG (model=$MODEL bg_load=$BG_LOAD)"
[ "$HANGS" -eq 0 ] && { echo "PASS"; exit 0; } || { echo "FAIL"; exit 1; }
