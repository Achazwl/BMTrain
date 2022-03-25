MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=$1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} $2"
echo ${CMD}
${CMD}
# usage: bash test_dist.sh 2 test_lamb.py