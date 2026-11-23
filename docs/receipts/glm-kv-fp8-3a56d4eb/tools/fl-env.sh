# GLM-5.3-Flash FP4-Local at TP=2 over the hosts' direct link; sourced by rank.sh on both hosts.
export TF_GLM_WORLD=2 NCCL_COMM_ID=192.0.2.1:29571 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1
export LD_LIBRARY_PATH=$HOME/fl-receipt/nccl
GLM=$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5
PORT=8090
