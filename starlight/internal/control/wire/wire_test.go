package wire

import (
	"strings"
	"testing"
)

func TestArtifactValidate(t *testing.T) {
	good := []Artifact{
		{Kind: "hf", Repo: "TensorFold/Qwen3.8-27B-MLX-4bit", Revision: strings.Repeat("a", 40)},
		{Kind: "path", Path: "/models/qwen"},
		{Kind: "url", URL: "https://mirror.example/q.tar", SHA256: strings.Repeat("0", 64)},
	}
	for _, a := range good {
		if err := a.Validate(); err != nil {
			t.Errorf("%+v: %v", a, err)
		}
	}
	bad := []Artifact{
		{Kind: "hf", Repo: "TensorFold/Qwen", Revision: "main"},
		{Kind: "hf", Repo: "../etc", Revision: strings.Repeat("a", 40)},
		{Kind: "path", Path: "models/q"},
		{Kind: "path", Path: "/models/../etc"},
		{Kind: "url", URL: "http://mirror.example/q.tar", SHA256: strings.Repeat("0", 64)},
		{Kind: "url", URL: "https://mirror.example/q.tar"},
		{Kind: "shell", Path: "/bin/sh"},
	}
	for _, a := range bad {
		if err := a.Validate(); err == nil {
			t.Errorf("accepted %+v", a)
		}
	}
}

func TestEngineRouteAllowlist(t *testing.T) {
	ok := [][2]string{{"POST", "/v1/chat/completions"}, {"POST", "/v1/messages"}, {"GET", "/v1/responses/resp_abc"}, {"DELETE", "/v1/responses/resp_abc"}}
	for _, r := range ok {
		if !AllowedEngineRoute(r[0], r[1]) {
			t.Errorf("refused %v", r)
		}
	}
	no := [][2]string{{"GET", "/metrics"}, {"POST", "/v1/slide/learn"}, {"DELETE", "/v1/imprint"}, {"GET", "/v1/responses/../x"}, {"POST", "/v1/responses/abc"}, {"GET", "/memory"}}
	for _, r := range no {
		if AllowedEngineRoute(r[0], r[1]) {
			t.Errorf("allowed %v", r)
		}
	}
}

func TestSpecBounds(t *testing.T) {
	s := Spec{Model: Artifact{Kind: "path", Path: "/m"}, Context: 32768, Parallel: 8, KVCache: "fp8"}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	s.Parallel = 1000
	if s.Validate() == nil {
		t.Fatal("accepted parallel 1000")
	}
	s.Parallel, s.Backend = 1, "rocm-evil"
	if s.Validate() == nil {
		t.Fatal("accepted unknown backend")
	}
}

func TestTensorParallelNeedsAScheme(t *testing.T) {
	base := Spec{Model: Artifact{Kind: "path", Path: "/m"}}
	for _, c := range []struct {
		s  Spec
		ok bool
	}{
		{Spec{GPUs: 1}, true}, {Spec{GPUs: 2}, false}, {Spec{GPUs: 4, TensorParallel: "glm"}, true},
		{Spec{GPUs: 2, TensorParallel: "nemotron", Backend: "cuda"}, true}, {Spec{GPUs: 1, TensorParallel: "glm"}, false},
		{Spec{GPUs: 2, TensorParallel: "glm", Backend: "metal"}, false}, {Spec{GPUs: 2, TensorParallel: "mpi"}, false},
		{Spec{GPUs: 17, TensorParallel: "glm"}, false},
	} {
		s := c.s
		s.Model = base.Model
		if err := s.Validate(); (err == nil) != c.ok {
			t.Errorf("%+v: validate = %v", c.s, err)
		}
	}
}

func TestGroupsAcrossNodes(t *testing.T) {
	m := Artifact{Kind: "path", Path: "/m"}
	for _, c := range []struct {
		s  Spec
		ok bool
	}{
		{Spec{Nodes: 2, TensorParallel: "glm"}, true}, {Spec{Nodes: 2}, false}, {Spec{Nodes: 2, GPUs: 4, TensorParallel: "rocm"}, true},
		{Spec{Nodes: 3, TensorParallel: "rocm"}, false}, {Spec{Nodes: 2, TensorParallel: "deepseek", Backend: "cuda"}, true},
		{Spec{Nodes: 2, TensorParallel: "minimax", Backend: "hip"}, false}, {Spec{Nodes: 17, TensorParallel: "glm"}, false},
	} {
		c.s.Model = m
		if err := c.s.Validate(); (err == nil) != c.ok {
			t.Errorf("%+v: %v", c.s, err)
		}
	}
	s := Spec{Model: m, Nodes: 2, GPUs: 4, TensorParallel: "glm"}
	good := Group{Member: 1, World: 8, FirstRank: 4, Master: "10.0.0.1", Port: 29500}
	if err := good.Validate(s); err != nil {
		t.Fatal(err)
	}
	for _, g := range []Group{{Member: 1, World: 8, FirstRank: 1, Master: "10.0.0.1", Port: 29500}, {Member: 2, World: 8, FirstRank: 8, Master: "h", Port: 29500},
		{Member: 1, World: 8, FirstRank: 4, Master: "bad host;rm", Port: 29500}, {Member: 1, World: 8, FirstRank: 4, Master: "gpu-2.example", Port: 80}} {
		if g.Validate(s) == nil {
			t.Errorf("accepted %+v", g)
		}
	}
	if (Group{Member: 0, World: 4, Master: "10.0.0.1", Port: 29500}).Validate(Spec{Model: m, GPUs: 4, TensorParallel: "glm"}) == nil {
		t.Error("accepted a rank group for a replica on one node")
	}
}
