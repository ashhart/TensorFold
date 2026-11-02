package e2e_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// spark presents a node as a GB10: CUDA on one GPU that shares the machine's memory, reachable for ranks on loopback.
func spark(context.Context) wire.Facts {
	return wire.Facts{Platform: "linux", Arch: "arm64", Backend: "cuda", MemoryBytes: 128 << 30, GPUs: []wire.GPU{{Name: "NVIDIA GB10"}},
		RankHost: "127.0.0.1"}
}

func TestTensorParallelGroupAcrossTwoNodes(t *testing.T) {
	f := startStarlight(t, nil)
	exec(t, f.pool, `UPDATE deployments SET spec = spec || '{"nodes": 2, "tensor_parallel": "glm", "backend": "cuda", "memory_gib": 160}'::jsonb`)
	lead, follower := t.TempDir(), t.TempDir()
	f.enrolWith(f.nodeURL, "spark-1", 0, nodeOptions{facts: spark, env: []string{"FAKE_ENGINE_RANK_DIR=" + lead}})
	f.enrolWith(f.nodeURL, "spark-2", 1, nodeOptions{facts: spark, env: []string{"FAKE_ENGINE_RANK_DIR=" + follower}})
	var members int
	_ = f.pool.QueryRow(f.ctx, "SELECT count(*) FROM replica_members WHERE state = 'ready'").Scan(&members)
	if members != 1 {
		t.Fatalf("%d ready members; the second node should run rank 1", members)
	}
	rank := func(dir, n string) map[string]string {
		var m map[string]string
		matches, _ := filepath.Glob(filepath.Join(dir, "rank"+n+".json"))
		if len(matches) == 0 {
			matches, _ = filepath.Glob(filepath.Join(lead, "rank"+n+".json"))
			matches2, _ := filepath.Glob(filepath.Join(follower, "rank"+n+".json"))
			matches = append(matches, matches2...)
		}
		if len(matches) == 0 {
			t.Fatalf("no record of rank %s", n)
		}
		b, _ := os.ReadFile(matches[0])
		_ = json.Unmarshal(b, &m)
		return m
	}
	r0, r1 := rank(lead, "0"), rank(follower, "1")
	if r0["world"] != "2" || r1["world"] != "2" || r0["comm"] != r1["comm"] || r0["comm"] != "127.0.0.1:29500" {
		t.Fatalf("ranks met at %v and %v", r0, r1)
	}
	resp, body := f.call("POST", "/v1/chat/completions", f.key, `{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("chat through a two-node group = %d %s", resp.StatusCode, body)
	}
	pid, _ := strconv.Atoi(r1["pid"])
	if p, err := os.FindProcess(pid); err == nil {
		_ = p.Kill()
	}
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		resp, _ := f.call("POST", "/v1/chat/completions", f.key, `{"model":"qwen3.8-27b","messages":[{"role":"user","content":"again"}]}`)
		if again := rank(follower, "1"); resp.StatusCode == 200 && again["pid"] != r1["pid"] {
			return
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatal("the group did not come back after its follower rank died")
}
