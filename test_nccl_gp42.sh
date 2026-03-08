#!/bin/bash
# =============================================================================
# NCCL Configuration Benchmark for gp42
# Tests all-reduce of 1GB bf16 tensor across 7x RTX PRO 6000 Blackwell GPUs
# GPU0 (A100) is excluded; only GPU1-7 (Blackwell) are used.
#
# Topology (from nvidia-smi topo -m):
#   GPU1-3: NUMA 0, interconnected via PXB/PIX
#   GPU4-7: NUMA 1, interconnected via PXB/PIX
#   Cross-NUMA: SYS (PCIe + SMP/QPI interconnect)
#
# Key findings from 24GB round:
#   - Tree (4175ms) >> Ring (4822ms) — 13% faster
#   - P2P_LEVEL=SYS with Tree,Ring was WORSE (5358ms)
#   - P2P disabled was terrible (7610ms)
#   - Default P2P level is optimal for most configs
# =============================================================================
set -e

# Only use the 7 Blackwell GPUs
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NUM_GPUS=7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/nccl_bench.py"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"

if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
fi

LOGDIR="$SCRIPT_DIR/nccl_bench_logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"
echo "" > "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo " NCCL Benchmark on gp42 — 7x RTX PRO 6000 Blackwell 96GB"   | tee -a "$SUMMARY"
echo " Payload: 1 GB bf16 all-reduce"                              | tee -a "$SUMMARY"
echo " Date: $(date)"                                              | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# ---- Configuration grid ----
# Each config is:  "LABEL|NCCL_ALGO|NCCL_MIN_NCHANNELS|NCCL_PROTO|NCCL_IB_PCI_RELAXED_ORDERING"
CONFIGS=(
    # --- Best from previous run ---
    "best_base|Tree|8||"

    # --- Test Protocols (1GB payload is "large", so SIMPLE might be better) ---
    "proto_SIMPLE|Tree|8|Simple|"
    "proto_LL|Tree|8|LL|"
    
    # --- Test PCIe Relaxed Ordering (often helps cross-NUMA PCIe) ---
    "relaxed_ord_1|Tree|8||1"
    "relaxed_ord_1_SIMPLE|Tree|8|Simple|1"

    # --- Double check higher channels ---
    "chan_16_base|Tree|16||"
    "chan_16_SIMPLE|Tree|16|Simple|1"
)

run_bench() {
    local label="$1"
    local algo="$2"
    local min_nchannels="$3"
    local proto="$4"
    local relaxed_ord="$5"

    # Unset related NCCL vars first
    unset NCCL_ALGO NCCL_MIN_NCHANNELS NCCL_PROTO NCCL_IB_PCI_RELAXED_ORDERING

    # Set only specified vars
    [ -n "$algo" ]          && export NCCL_ALGO="$algo"
    [ -n "$min_nchannels" ] && export NCCL_MIN_NCHANNELS="$min_nchannels"
    [ -n "$proto" ]         && export NCCL_PROTO="$proto"
    [ -n "$relaxed_ord" ]   && export NCCL_IB_PCI_RELAXED_ORDERING="$relaxed_ord"

    export NCCL_DEBUG=WARN

    local logfile="$LOGDIR/${label}_v2.log"
    echo "" | tee -a "$SUMMARY"
    echo ">>> [$label] Starting..." | tee -a "$SUMMARY"
    echo "    ALGO=${algo:-default} MIN_NCHANNELS=${min_nchannels:-default} PROTO=${proto:-default} RELAXED_ORD=${relaxed_ord:-default}" | tee -a "$SUMMARY"

    # Run benchmark
    if torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS "$BENCH_SCRIPT" > "$logfile" 2>&1; then
        # Extract result
        local min_ms=$(grep "\[RESULT\] Min:" "$logfile" | awk '{print $3}')
        local avg_ms=$(grep "\[RESULT\] Avg:" "$logfile" | awk '{print $3}')
        local algo_bw=$(grep "\[RESULT\] Algo BW" "$logfile" | awk '{print $5}')
        local bus_bw=$(grep "\[RESULT\] Bus  BW" "$logfile" | awk '{print $5}')
        echo "    ✓ Min: ${min_ms} ms | Avg: ${avg_ms} ms | Algo BW: ${algo_bw} GB/s | Bus BW: ${bus_bw} GB/s" | tee -a "$SUMMARY"
    else
        echo "    ✗ FAILED (see $logfile)" | tee -a "$SUMMARY"
    fi
}

# Run all configurations
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r label algo p2p_level p2p_disable buffsize nthreads min_nchannels shm_disable extra <<< "$config"
    run_bench "$label" "$algo" "$p2p_level" "$p2p_disable" "$buffsize" "$nthreads" "$min_nchannels" "$shm_disable" "$extra"
done

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo " ALL TESTS COMPLETE" | tee -a "$SUMMARY"
echo " Summary saved to: $SUMMARY" | tee -a "$SUMMARY"
echo " Individual logs in: $LOGDIR/" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# Sort results by min time
echo "" | tee -a "$SUMMARY"
echo "--- RANKING (by min latency) ---" | tee -a "$SUMMARY"
grep "✓" "$SUMMARY" | sed 's/.*✓ //' | sort -t'|' -k1 -n | while read line; do
    config_label=$(grep -B3 "$line" "$SUMMARY" | grep ">>>" | tail -1 | sed 's/.*\[//;s/\].*//')
    echo "  [$config_label] $line"
done | tee -a "$SUMMARY"
