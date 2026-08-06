#!/bin/bash
# ────────────────────────────────────────────────────────────
# AstraKV-W Edge Device Simulator Setup
# ────────────────────────────────────────────────────────────
# Creates a cgroup v2 memory constraint to simulate
# edge/embedded device memory limits for competition
# Task 2 (virtual memory techniques) validation.
#
# Usage:
#   sudo bash scripts/archive/setup_edge_sim.sh [mem_limit_gb]
#
# Example:
#   sudo bash scripts/archive/setup_edge_sim.sh 16   # Simulate 16 GB edge node
#   sudo bash scripts/archive/setup_edge_sim.sh 24   # Simulate 24 GB edge node
# ────────────────────────────────────────────────────────────
set -euo pipefail

CGROUP_NAME="${CGROUP_NAME:-astrakv_edge_sim}"
CGROUP_PATH="/sys/fs/cgroup/$CGROUP_NAME"
MEM_LIMIT_GB="${1:-16}"

# ── check prerequisites ──
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script requires root privileges for cgroup setup."
    echo "Run: sudo bash scripts/archive/setup_edge_sim.sh [mem_gb]"
    exit 1
fi

if [ ! -d /sys/fs/cgroup ]; then
    echo "ERROR: cgroup filesystem not mounted at /sys/fs/cgroup"
    exit 1
fi

# ── detect cgroup version ──
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    CGROUP_VER="v2"
elif [ -d /sys/fs/cgroup/memory ]; then
    CGROUP_VER="v1"
    CGROUP_PATH="/sys/fs/cgroup/memory/$CGROUP_NAME"
else
    echo "ERROR: Unsupported cgroup configuration"
    exit 1
fi

echo "Detected cgroup $CGROUP_VER"

# ── create cgroup ──
if [ ! -d "$CGROUP_PATH" ]; then
    mkdir -p "$CGROUP_PATH"
    echo "Created cgroup: $CGROUP_PATH"
else
    echo "Cgroup already exists: $CGROUP_PATH"
fi

# ── set memory limit ──
MEM_BYTES=$((MEM_LIMIT_GB * 1024 * 1024 * 1024))

if [ "$CGROUP_VER" = "v2" ]; then
    # cgroup v2
    echo "$MEM_BYTES" > "$CGROUP_PATH/memory.max"

    # Disable swap to simulate real edge devices without swap
    if [ -f "$CGROUP_PATH/memory.swap.max" ]; then
        echo 0 > "$CGROUP_PATH/memory.swap.max"
    fi

    echo "Memory limit: ${MEM_LIMIT_GB} GB (${MEM_BYTES} bytes)"
    echo "Swap: disabled"

    # ── show current state ──
    echo ""
    echo "=== cgroup $CGROUP_NAME status ==="
    echo "memory.max:      $(cat "$CGROUP_PATH/memory.max")"
    echo "memory.current:  $(cat "$CGROUP_PATH/memory.current" 2>/dev/null || echo 'N/A')"
    echo "memory.swap.max: $(cat "$CGROUP_PATH/memory.swap.max" 2>/dev/null || echo 'N/A')"
else
    # cgroup v1
    echo "$MEM_BYTES" > "$CGROUP_PATH/memory.limit_in_bytes"
    echo 0 > "$CGROUP_PATH/memory.swappiness" 2>/dev/null || true
    echo "Memory limit: ${MEM_LIMIT_GB} GB (${MEM_BYTES} bytes)"
fi

echo ""
echo "=== Next steps ==="
echo "To run processes inside this cgroup:"
echo "  sudo cgexec -g memory:$CGROUP_NAME bash"
echo ""
echo "Or add current shell to cgroup:"
echo "  echo \$\$ | sudo tee $CGROUP_PATH/cgroup.procs"
echo ""
echo "To clean up:"
echo "  sudo rmdir $CGROUP_PATH"
