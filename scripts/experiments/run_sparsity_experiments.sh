#!/bin/bash

################################################################################
# Sparsity-Aware Natural Niches Experiment Runner for SparseFusion
################################################################################
# 
# This script runs multiple experiments comparing different omega/beta
# configurations to explore the Pareto frontier between performance and sparsity.
#
################################################################################

# Configuration
TOTAL_FP=1000           # Total forward passes per experiment
RUNS=1                  # Number of independent runs
POP_SIZE=10             # Population size
MODEL1="models/Qwen2.5-Math-1.5B-Instruct"
MODEL2="models/Qwen2.5-Coder-1.5B-Instruct"
ARCHIVE_BACKEND="gpu"   # gpu or cpu

# Create experiment directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="experiments/sparsity_comparison_${TIMESTAMP}"
mkdir -p "${EXP_DIR}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Sparsity-Aware Natural Niches Experiment Suite               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Experiment directory: ${EXP_DIR}"
echo "🔢 Configuration:"
echo "   - Total forward passes: ${TOTAL_FP}"
echo "   - Runs per experiment: ${RUNS}"
echo "   - Population size: ${POP_SIZE}"
echo "   - Archive backend: ${ARCHIVE_BACKEND}"
echo ""
echo "🧪 Experiments to run:"
echo "   1. Baseline (Original Natural Niches - no sparsity awareness)"
echo "   2. Balanced (ω=0.5, β=0.5)"
echo "   3. Performance-Focused (ω=0.8, β=0.2)"
echo "   4. Sparsity-Focused (ω=0.2, β=0.8)"
echo "   5. Extreme Sparsity (ω=0.1, β=0.9)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Array to store background process PIDs
declare -a PIDS

# Experiment 1: Baseline (Original Natural Niches)
echo "🚀 [1/5] Starting Baseline (Original Natural Niches)..."
python main.py \
    --pop_size ${POP_SIZE} \
    --total_forward_passes ${TOTAL_FP} \
    --runs ${RUNS} \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --archive_backend ${ARCHIVE_BACKEND} \
    > "${EXP_DIR}/baseline.log" 2>&1 &
PIDS[0]=$!
echo "   PID: ${PIDS[0]}"

# Move result file after completion
(
    wait ${PIDS[0]}
    if [ -f "results/natural_niches.pkl" ]; then
        mv "results/natural_niches.pkl" "${EXP_DIR}/baseline.pkl"
        echo "✅ [1/5] Baseline completed!"
    fi
) &

sleep 2

# Experiment 2: Balanced (ω=0.5, β=0.5)
echo "🚀 [2/5] Starting Balanced (ω=0.5, β=0.5)..."
python main_natural_niches_sparsity_aware_fn.py \
    --pop_size ${POP_SIZE} \
    --total_forward_passes ${TOTAL_FP} \
    --runs ${RUNS} \
    --omega 0.5 \
    --beta 0.5 \
    --tau 1.0 \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --archive_backend ${ARCHIVE_BACKEND} \
    --output_dir "${EXP_DIR}" \
    > "${EXP_DIR}/balanced.log" 2>&1 &
PIDS[1]=$!
echo "   PID: ${PIDS[1]}"

sleep 2

# Experiment 3: Performance-Focused (ω=0.8, β=0.2)
echo "🚀 [3/5] Starting Performance-Focused (ω=0.8, β=0.2)..."
python main_natural_niches_sparsity_aware_fn.py \
    --pop_size ${POP_SIZE} \
    --total_forward_passes ${TOTAL_FP} \
    --runs ${RUNS} \
    --omega 0.8 \
    --beta 0.2 \
    --tau 1.0 \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --archive_backend ${ARCHIVE_BACKEND} \
    --output_dir "${EXP_DIR}" \
    > "${EXP_DIR}/performance_focused.log" 2>&1 &
PIDS[2]=$!
echo "   PID: ${PIDS[2]}"

sleep 2

# Experiment 4: Sparsity-Focused (ω=0.2, β=0.8)
echo "🚀 [4/5] Starting Sparsity-Focused (ω=0.2, β=0.8)..."
python main_natural_niches_sparsity_aware_fn.py \
    --pop_size ${POP_SIZE} \
    --total_forward_passes ${TOTAL_FP} \
    --runs ${RUNS} \
    --omega 0.2 \
    --beta 0.8 \
    --tau 1.0 \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --archive_backend ${ARCHIVE_BACKEND} \
    --output_dir "${EXP_DIR}" \
    > "${EXP_DIR}/sparsity_focused.log" 2>&1 &
PIDS[3]=$!
echo "   PID: ${PIDS[3]}"

sleep 2

# Experiment 5: Extreme Sparsity (ω=0.1, β=0.9)
echo "🚀 [5/5] Starting Extreme Sparsity (ω=0.1, β=0.9)..."
python main_natural_niches_sparsity_aware_fn.py \
    --pop_size ${POP_SIZE} \
    --total_forward_passes ${TOTAL_FP} \
    --runs ${RUNS} \
    --omega 0.1 \
    --beta 0.9 \
    --tau 0.5 \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --archive_backend ${ARCHIVE_BACKEND} \
    --output_dir "${EXP_DIR}" \
    > "${EXP_DIR}/extreme_sparsity.log" 2>&1 &
PIDS[4]=$!
echo "   PID: ${PIDS[4]}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ ⏳ Waiting for all experiments to complete (5 tasks)           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "💡 Monitoring tips:"
echo "   - View all logs: ls -lh ${EXP_DIR}/*.log"
echo "   - Monitor specific log: tail -f ${EXP_DIR}/<experiment>.log"
echo "   - Check running processes: ps aux | grep python | grep main"
echo ""
echo "📋 Experiment list:"
echo "   [1/5] Baseline - PID ${PIDS[0]}"
echo "   [2/5] Balanced - PID ${PIDS[1]}"
echo "   [3/5] Performance-Focused - PID ${PIDS[2]}"
echo "   [4/5] Sparsity-Focused - PID ${PIDS[3]}"
echo "   [5/5] Extreme Sparsity - PID ${PIDS[4]}"
echo ""

# Wait for all experiments to complete
for i in {0..4}; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Experiment $((i+1))/5 completed successfully"
    else
        echo "❌ Experiment $((i+1))/5 failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ ✅ All experiments completed!                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results saved in: ${EXP_DIR}/"
echo ""
echo "📋 Generated files:"
ls -lh "${EXP_DIR}"/*.pkl 2>/dev/null || echo "   (No .pkl files found)"
echo ""
echo "📈 Next steps:"
echo "   1. Analyze results: python analyze_sparsity_experiments.py ${EXP_DIR}"
echo "   2. Visualize: python plot_sparsity_comparison.py ${EXP_DIR}"
echo ""
echo "════════════════════════════════════════════════════════════════"

