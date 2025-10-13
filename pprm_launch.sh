#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

ACTION=${1:-start}
MODEL_TYPE=${2:-"pprm_3b_data"}
PORT=${3:-30000}
DP_SIZE=${4:-1}
TP_SIZE=${5:-4}
PP_SIZE=${6:-1}
MEM_FRACTION=${7:-"0.8"}

MODEL_PATH=""

# Model path configuration - focusing on reward models
case $MODEL_TYPE in
    pprm_3b_data)
        MODEL_PATH="model_path_to_pprm_3b_model"
        ;;
    pprm_7b_data)
        MODEL_PATH="model_path_to_pprm_7b_model"
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        echo "Supported models: pprm_3b_data, pprm_7b_data"
        exit 1
        ;;
esac

case "$ACTION" in
    start)
        echo "Starting SGLang Router (model=$MODEL_TYPE) on port $PORT with TP_SIZE=$TP_SIZE"
        
        if [ "$DP_SIZE" -eq 1 ]; then
            echo "[INFO] Launching sglang server via sglang.launch_server"
            CMD="python -m sglang.launch_server \
                --model-path $MODEL_PATH \
                --port $PORT \
                --tp $TP_SIZE \
                --pp-size $PP_SIZE \
                --max-running-requests 2048 \
                --chunked-prefill-size 8192 \
                --mem-fraction-static $MEM_FRACTION"
        else
            echo "[INFO] Launching sglang server via sglang_router.launch_server"
            CMD="python -m sglang_router.launch_server \
                --model-path \"$MODEL_PATH\" \
                --port $PORT \
                --dp-size $DP_SIZE \
                --tp-size $TP_SIZE \
                --pp-size $PP_SIZE \
                --router-worker-startup-timeout-secs 1000 \
                --mem-fraction-static $MEM_FRACTION"
        fi
        
        echo "Command: $CMD"
        eval $CMD
        echo "Router launched at http://127.0.0.1:$PORT"
        ;;
    stop)
        echo "[INFO] Stopping all SGLang processes..."
        pkill -f "sglang_router.launch_server"
        pkill -f "sglang.launch_server"
        pkill -f "sglang_router"
        pkill -f "sglang"
        sleep 2
        pkill -9 -f "sglang_router" 2>/dev/null
        pkill -9 -f "sglang" 2>/dev/null
        echo "[INFO] Stop completed!"
        ;;
    *)
        echo "Usage:"
        echo "  $0 start <model_type> <port> <dp_size> <tp_size> <pp_size>"
        echo ""
        echo "Parameters:"
        echo "  model_type  - RM model identifier (rm, skywork, etc.)"
        echo "  port        - Server port (default: 30000)"
        echo "  dp_size     - Data parallelism size (default: 1)"
        echo "  tp_size     - Tensor parallelism size (default: 1)"
        echo "  pp_size     - Pipeline parallelism size (default: 1)"
        echo ""
        echo "Example:"
        echo "  $0 start rm 30000 1 4 1  # Start rm model on port 30000 with TP_SIZE=4"
        exit 1
        ;;
esac