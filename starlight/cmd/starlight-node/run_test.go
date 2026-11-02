package main

import "testing"

func TestEngineEnvAllowlist(t *testing.T) {
	for v, ok := range map[string]bool{"TF_GLM_LAYERS=40": true, "NCCL_SOCKET_IFNAME=ens3": true, "TENSORFOLD_MEMORY_RESERVE_GIB=8": true,
		"CUDA_VISIBLE_DEVICES=0,1": true, "HIP_VISIBLE_DEVICES=1": true, "ROCR_VISIBLE_DEVICES=0": true, "NCCL_COMM_ID=10.0.0.1:1": false, "TENSORFOLD_API_KEY=x": false, "PATH=/tmp": false,
		"LD_PRELOAD=/x.so": false, "NCCL_DEBUG": false} {
		var e envList
		if err := e.Set(v); (err == nil) != ok {
			t.Errorf("--env %s: %v", v, err)
		}
	}
}
