package engine

import (
	"slices"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

func envOf(r rankSpec, key string) string {
	for i := len(r.env) - 1; i >= 0; i-- {
		if v, ok := strings.CutPrefix(r.env[i], key+"="); ok {
			return v
		}
	}
	return ""
}

func TestRankLayoutsPerScheme(t *testing.T) {
	base := wire.Assignment{Spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}}}
	glm := base
	glm.Spec.GPUs, glm.Spec.Nodes, glm.Spec.TensorParallel = 2, 2, "glm"
	glm.Devices = []string{"GPU-aaaa", "GPU-bbbb"}
	glm.Group = &wire.Group{Member: 1, World: 4, FirstRank: 2, Master: "10.0.0.1", Port: 29510}
	ranks := rankSpecs(glm, []string{"serve", "/m"}, nil, 0)
	if len(ranks) != 2 || ranks[0].rank != 2 || ranks[1].rank != 3 || envOf(ranks[1], "TF_GLM_RANK") != "3" || envOf(ranks[1], "TF_GLM_WORLD") != "4" ||
		envOf(ranks[0], "NCCL_COMM_ID") != "10.0.0.1:29510" || envOf(ranks[1], "CUDA_VISIBLE_DEVICES") != "GPU-bbbb" {
		t.Fatalf("GLM member ranks %+v", ranks)
	}
	ds := base
	ds.Spec.Nodes, ds.Spec.TensorParallel = 2, "deepseek"
	ds.Group = &wire.Group{Member: 0, World: 2, Master: "10.0.0.1", Port: 29500}
	r := rankSpecs(ds, []string{"serve", "/m"}, nil, 0)
	if len(r) != 1 || envOf(r[0], "TF_TP_MASTER") != "10.0.0.1" || envOf(r[0], "TF_TP_PORT") != "29500" || envOf(r[0], "TF_TP_RANK") != "0" {
		t.Fatalf("DeepSeek rank %+v", r)
	}
	rocm := base
	rocm.Spec.GPUs, rocm.Spec.TensorParallel, rocm.Spec.Backend = 4, "rocm", "hip"
	rocm.Devices = []string{"0", "1", "2", "3"}
	r = rankSpecs(rocm, []string{"serve", "/m"}, nil, 29600)
	want := []string{"--device", "3", "--tp", "4", "--rank", "3", "--master", "127.0.0.1", "--master-port", "29600"}
	if len(r) != 4 || envOf(r[3], "HIP_VISIBLE_DEVICES") != "0,1,2,3" || !slices.Equal(r[3].args[2:], want) {
		t.Fatalf("ROCm rank 3 %+v", r[3])
	}
	if err := checkDevices(glm); err != nil {
		t.Fatalf("a valid member refused: %v", err)
	}
	glm.Group = nil
	if checkDevices(glm) == nil {
		t.Fatal("a replica spanning nodes was accepted without its group")
	}
}

// hipList reads a device list as HIP does: HIP_VISIBLE_DEVICES when set, else CUDA_VISIBLE_DEVICES.
func hipList(r rankSpec) string {
	for i := len(r.env) - 1; i >= 0; i-- {
		if v, ok := strings.CutPrefix(r.env[i], "HIP_VISIBLE_DEVICES="); ok {
			return v
		}
	}
	return envOf(r, "CUDA_VISIBLE_DEVICES")
}

func TestPinsOverrideTheAgentsOwnDeviceLists(t *testing.T) {
	inherited := []string{"PATH=/usr/bin", "HIP_VISIBLE_DEVICES=4,5,6,7", "CUDA_VISIBLE_DEVICES=4,5,6,7"}
	for _, backend := range []string{"hip", "cuda", ""} {
		a := wire.Assignment{Spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, GPUs: 2, TensorParallel: "rocm", Backend: backend},
			Devices: []string{"GPU-4a5b6c7d8e9fa0b1", "GPU-5a6b7c8d9eafb0c1"}}
		if backend == "cuda" {
			a.Spec.TensorParallel = "glm"
		}
		for _, r := range rankSpecs(a, []string{"serve", "/m"}, inherited, 29600) {
			want := a.Devices[r.rank]
			if a.Spec.TensorParallel == "rocm" {
				want = strings.Join(a.Devices, ",")
			}
			if hipList(r) != want || envOf(r, "CUDA_VISIBLE_DEVICES") != want {
				t.Errorf("backend %q rank %d: HIP sees %q and CUDA %q, want %q", backend, r.rank, hipList(r), envOf(r, "CUDA_VISIBLE_DEVICES"), want)
			}
		}
		one := wire.Assignment{Spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, Backend: backend}, Devices: []string{"GPU-6a7b8c9daebfc0d1"}}
		if r := rankSpecs(one, []string{"serve", "/m"}, inherited, 0)[0]; hipList(r) != "GPU-6a7b8c9daebfc0d1" || envOf(r, "CUDA_VISIBLE_DEVICES") != "GPU-6a7b8c9daebfc0d1" {
			t.Errorf("backend %q: a one-GPU replica sees HIP %q and CUDA %q", backend, hipList(r), envOf(r, "CUDA_VISIBLE_DEVICES"))
		}
	}
}
