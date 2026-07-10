#!/bin/bash
# bench-caffeinated.sh -- run a benchmark command with machine-state validity guards.
#
# Motivation (2026-07-10): an overnight campaign lost 6 of 9 runs to clamshell
# sleep -- the machine slept 18 minutes in, and every later run executed in
# ~2-minute DarkWake slivers per 15-minute sleep cycle, guaranteeing wall-clock
# timeouts and contaminating one run's timings. Benchmarks are only valid on an
# awake machine on AC power.
#
# Usage: scripts/bench-caffeinated.sh <logfile> <command...>
#   - refuses to start on battery power (AC required);
#   - runs the command under `caffeinate -dims` (blocks display/idle/disk/system
#     sleep; NOTE: caffeinate does not reliably block clamshell sleep -- keep the
#     lid open or attach an external display for unattended campaigns);
#   - after the run, scans `pmset -g log` for Sleep entries inside the run
#     window and marks the run VOID in the logfile if any occurred.
set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <logfile> <command...>" >&2
  exit 2
fi
LOG=$1
shift

if ! pmset -g batt | grep -q "AC Power"; then
  echo "REFUSED: not on AC power (benchmarks on battery are invalid)" | tee -a "$LOG" >&2
  exit 3
fi

START_TS=$(date "+%Y-%m-%d %H:%M:%S")
echo "BENCH START $START_TS :: $*" >> "$LOG"

caffeinate -dims "$@"
STATUS=$?

END_TS=$(date "+%Y-%m-%d %H:%M:%S")
echo "BENCH END   $END_TS exit=$STATUS" >> "$LOG"

# Any 'Entering Sleep' event between START and END voids the run.
SLEEPS=$(pmset -g log | awk -v s="$START_TS" -v e="$END_TS" \
  '$0 ~ /Entering Sleep/ { ts = $1 " " $2; if (ts >= s && ts <= e) print }' | wc -l | tr -d ' ')
if [ "$SLEEPS" -gt 0 ]; then
  echo "BENCH VOID  $SLEEPS sleep event(s) inside the run window -- timings invalid, rerun" \
    | tee -a "$LOG" >&2
  exit 4
fi
exit $STATUS
