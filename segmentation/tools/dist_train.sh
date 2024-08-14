#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
#WORK_DIR=$3
#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}
#PORT=${PORT:-29500}
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch \
#    --nnodes=$NNODES \
#    --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \
#    --master_port=$PORT \
#    $(dirname "$0")/train.py \
#    $CONFIG \
#    --work_dir=$WORK_DIR \
#    --launcher pytorch ${@:3}

CONFIG=$1
WORK_DIR=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch  --nproc_per_node=2 --master_port=$((RANDOM + 10000)) \
    $(dirname "$0")/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:3}