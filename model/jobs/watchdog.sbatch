#!/bin/bash
#SBATCH --job-name=scrm-watchdog
# No --partition/--account directive here: pass them on the command line.
#   sbatch --partition=<p> --account=<a> model/jobs/watchdog.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:1G
set -uo pipefail
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PY=${SCRM_PY:-python3}
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py
cd "$REPO"
timeout 7.5h $PY measurements/watchdog.py --loop 600 --since 40
ST=$(sacct -j "$SLURM_JOB_ID" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
# Carry this job's own placement to the next hop, read from scontrol.
ACCT=${SLURM_JOB_ACCOUNT:-$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'Account=[^ ]+' | head -1 | cut -d= -f2)}
PART=${SLURM_JOB_PARTITION:-$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'Partition=[^ ]+' | head -1 | cut -d= -f2)}
case "$ST" in
    CANCELLED*) echo "cancelled; not resubmitting" ;;
    *) if [ -n "$ACCT" ] && [ -n "$PART" ]; then
           sbatch --account="$ACCT" --partition="$PART" "$REPO/model/jobs/watchdog.sbatch"
       else
           >&2 echo "could not read this job's account/partition; not resubmitting blind"
       fi ;;
esac
