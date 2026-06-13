#!/usr/bin/env bash
# run_multi_instance.sh
#
# Usage:
#   bash scripts/run_multi_instance.sh <sim_name> <n_instances> [options]
#
# Required:
#   sim_name      Tangos simulation name, e.g. Halo1459_DMO
#   n_instances   Number of independent DarkLight realisations
#
# Optional:
#   --ftag          Tagging fraction          (default: 0.01)
#   --halonumber    Halo number               (default: 1)
#   --no-mergers    Disable merger tagging
#   --output-prefix Output directory prefix   (default: <sim_name>_tagged)
#   --recursive     Use recursive tagging variant
#
# Examples:
#   bash scripts/run_multi_instance.sh Halo1459_DMO 50
#   bash scripts/run_multi_instance.sh Halo1459_DMO 50 --recursive
#   bash scripts/run_multi_instance.sh Halo1459_DMO 50 --ftag 0.005 --recursive

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <sim_name> <n_instances> [options]"
    exit 1
fi

SIM_NAME="$1"
N_INSTANCES="$2"
shift 2

FTAG="0.01"
HALONUMBER="1"
NO_MERGERS=""
OUTPUT_PREFIX="${SIM_NAME}_tagged"
RECURSIVE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ftag)          FTAG="$2";          shift 2 ;;
        --halonumber)    HALONUMBER="$2";    shift 2 ;;
        --no-mergers)    NO_MERGERS="--no-mergers"; shift ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --recursive)     RECURSIVE="--recursive";  shift ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

echo "============================================================"
echo "  sim        : ${SIM_NAME}"
echo "  n_instances: ${N_INSTANCES}"
echo "  ftag       : ${FTAG}"
echo "  halonumber : ${HALONUMBER}"
echo "  mergers    : ${NO_MERGERS:-enabled}"
echo "  prefix     : ${OUTPUT_PREFIX}"
echo "  recursive  : ${RECURSIVE:-no}"
echo "============================================================"

"$PYTHON" "${SCRIPT_DIR}/run_multi_instance.py" \
    "${SIM_NAME}" \
    --n-instances "${N_INSTANCES}" \
    --ftag "${FTAG}" \
    --halonumber "${HALONUMBER}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    ${RECURSIVE} \
    ${NO_MERGERS}

echo ""
echo "All done. Output: ${OUTPUT_PREFIX}/instance_*.csv"
