#!/usr/bin/env bash
# one.sh HOST NAME CMD [ARGS]: tf-cuda-test CMD on one node's GPU (container <container>), log ~/glm53-b/ct/NAME.log
set -u
H=$1; NAME=$2; shift 2
IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
ssh <user>@<lan>.$H "docker rm -f <container>$NAME >/dev/null 2>&1; docker run --rm --name <container>$NAME --gpus all --ipc host -v \$HOME/glm53-b/ct:/ct -w /ct --entrypoint sh $IMAGE -c '/ct/tf-cuda-test $* > /ct/$NAME.log 2>&1; echo exit=\$? >> /ct/$NAME.log'"
ssh <user>@<lan>.$H "grep -cE '^PASS' ~/glm53-b/ct/$NAME.log; grep -E 'FAIL|SUMMARY|exit=|glm-ref-dsa:' ~/glm53-b/ct/$NAME.log | tail -15"
