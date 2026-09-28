#!/usr/bin/env bash
# Run a long WSL job detached from the launching wsl.exe client.
#
# A process in the foreground of `wsl.exe` receives SIGHUP and dies when that
# client exits or is killed (for example by a tool timeout); WSL then idles
# the distro down ~15 s later, which looks like a WSL restart.  A
# `nohup setsid` child started through wsl.exe survives the client and keeps
# the distro alive until it exits.  (A `systemd-run` service does not: WSL's
# idle shutdown does not count it and kills it with the distro.)
#
# Usage:
#   wsl_run_detached.sh <job-name> <log-dir> <workdir> -- <command...>
# Then poll <log-dir>/<job-name>.status (absent while running; holds
# "exit=<code> end=<time>" when done) and <job-name>.{stdout,stderr}.log.
set -euo pipefail

if [[ $# -lt 5 || $4 != "--" ]]; then
  echo "usage: $0 <job-name> <log-dir> <workdir> -- <command...>" >&2
  exit 2
fi
job=$1 log_dir=$2 workdir=$3
shift 4
mkdir -p "$log_dir"
status_file=$log_dir/$job.status
rm -f "$status_file"

cd "$workdir"
# The job's own arguments follow "_" so the wrapper script needs no quoting.
nohup setsid bash -c '
  status_file=$1; shift
  echo "START $(date -Is) pid=$$" >&2
  "$@"
  status=$?
  echo "exit=$status end=$(date -Is)" > "$status_file"
  exit $status
' _ "$status_file" "$@" \
  >> "$log_dir/$job.stdout.log" 2>> "$log_dir/$job.stderr.log" < /dev/null &
pid=$!

# Exiting before the child has left this session lets the client teardown
# kill it, so wait until it has called setsid and logged START.
for _ in $(seq 1 100); do
  if [[ -e $status_file ]] || { [[ $(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ') == "$pid" ]] \
      && grep -q "^START .* pid=$pid$" "$log_dir/$job.stderr.log" 2>/dev/null; }; then
    echo "started $job pid=$pid logs=$log_dir"
    exit 0
  fi
  sleep 0.1
done
echo "failed to confirm $job start (pid=$pid)" >&2
exit 1
