package engine_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
)

func rankFile(t *testing.T, dir, rank string) map[string]string {
	t.Helper()
	var m map[string]string
	b, err := os.ReadFile(filepath.Join(dir, "rank"+rank+".json"))
	if err != nil || json.Unmarshal(b, &m) != nil {
		t.Fatalf("rank %s left no placement record: %v", rank, err)
	}
	return m
}

func TestSingleGPUReplicaSeesOnlyItsGPU(t *testing.T) {
	ranks := t.TempDir()
	sup, _ := newSup(t, func(c *engine.Config) { c.Env = []string{"FAKE_ENGINE_RANK_DIR=" + ranks} })
	id := ids.New()
	asg := nodetest.Assignment(id, "fake-27b", nodetest.ModelDir(t), 1, 1)
	asg.Devices = []string{"GPU-5e1f0000-0000-4000-8000-000000000003"}
	sup.Apply([]wire.Assignment{asg})
	waitFor(t, sup, 20*time.Second, state(id, wire.StateReady))
	if r := rankFile(t, ranks, "0"); r["visible"] != asg.Devices[0] || r["order"] != "PCI_BUS_ID" || r["comm"] != "" {
		t.Fatalf("engine environment %v", r)
	}
}

func TestRankGroupPinsEachRankAndFailsTogether(t *testing.T) {
	ranks := t.TempDir()
	sup, root := newSup(t, func(c *engine.Config) {
		c.Env = []string{"FAKE_ENGINE_RANK_DIR=" + ranks, "FAKE_ENGINE_FOLLOWER_CRASH_MS=3000"}
	})
	id := ids.New()
	asg := nodetest.Assignment(id, "fake-glm", nodetest.ModelDir(t), 4, 1)
	asg.Spec.GPUs, asg.Spec.TensorParallel = 2, "glm"
	asg.Devices = []string{"GPU-aaaa1111", "GPU-bbbb2222"}
	sup.Apply([]wire.Assignment{asg})
	waitFor(t, sup, 20*time.Second, state(id, wire.StateReady))
	lead, follower := rankFile(t, ranks, "0"), rankFile(t, ranks, "1")
	if lead["visible"] != "GPU-aaaa1111" || follower["visible"] != "GPU-bbbb2222" || lead["world"] != "2" || follower["world"] != "2" {
		t.Fatalf("ranks pinned %v and %v", lead, follower)
	}
	if !strings.HasPrefix(lead["comm"], "127.0.0.1:") || lead["comm"] != follower["comm"] {
		t.Fatalf("ranks meet at %q and %q", lead["comm"], follower["comm"])
	}
	if _, err := os.Stat(filepath.Join(root, id, "engine.rank1.log")); err != nil {
		t.Fatalf("follower log missing: %v", err)
	}
	first := pid(t, root, id)
	waitFor(t, sup, 15*time.Second, func(ss []wire.ReplicaStatus) bool {
		return len(ss) == 1 && ss[0].State != wire.StateReady && strings.Contains(ss[0].Error, "rank 1")
	})
	deadline := time.Now().Add(10 * time.Second)
	for alive(first) && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if alive(first) {
		t.Fatalf("rank 0 (pid %d) kept running after its follower died", first)
	}
}

func TestDevicesMustMatchTheSpec(t *testing.T) {
	asg := nodetest.Assignment(ids.New(), "m", "/m", 1, 1)
	for _, c := range []struct {
		gpus    int
		tp      string
		devices []string
	}{
		{2, "glm", nil}, {2, "glm", []string{"0"}}, {1, "", []string{"0", "1"}}, {2, "glm", []string{"0", "0"}}, {1, "", []string{"0;rm -rf"}},
	} {
		a := asg
		a.Spec.GPUs, a.Spec.TensorParallel, a.Devices = c.gpus, c.tp, c.devices
		if engine.Check(a) == nil {
			t.Errorf("accepted gpus %d devices %v", c.gpus, c.devices)
		}
	}
	asg.Devices = []string{"7"}
	if err := engine.Check(asg); err != nil {
		t.Fatalf("refused an index: %v", err)
	}
}
