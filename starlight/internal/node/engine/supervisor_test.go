package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/artifacts"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

func TestMain(m *testing.M) {
	code := m.Run()
	fakeengine.RemoveBinary()
	os.Exit(code)
}

func newSup(t *testing.T, mut func(*engine.Config)) (*engine.Supervisor, string) {
	t.Helper()
	bin := fakeengine.Binary(t)
	root := t.TempDir()
	cfg := engine.Config{Engine: bin, Root: root, Stager: &artifacts.Stager{Engine: bin, ModelsDir: t.TempDir()},
		StartupTimeout: 15 * time.Second, DrainTimeout: 3 * time.Second, StopGrace: 5 * time.Second,
		Backoff: func(int) time.Duration { return 100 * time.Millisecond }, CrashLimit: 3, CrashWindow: time.Minute}
	if mut != nil {
		mut(&cfg)
	}
	sup, err := engine.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		sup.Shutdown(ctx)
	})
	return sup, root
}

func waitFor(t *testing.T, sup *engine.Supervisor, timeout time.Duration, ok func([]wire.ReplicaStatus) bool) []wire.ReplicaStatus {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var last []wire.ReplicaStatus
	for time.Now().Before(deadline) {
		last = sup.Statuses()
		if ok(last) {
			return last
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("condition not met within %s; statuses %+v", timeout, last)
	return nil
}

func state(id, want string) func([]wire.ReplicaStatus) bool {
	return func(ss []wire.ReplicaStatus) bool {
		for _, s := range ss {
			if s.ReplicaID == id && s.State == want {
				return true
			}
		}
		return false
	}
}

func pid(t *testing.T, root, id string) int {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, id, "engine.pid"))
	if err != nil {
		return 0
	}
	n, _ := strconv.Atoi(strings.SplitN(string(b), "\n", 2)[0])
	return n
}

func TestLifecycleServeResizeRestartRemove(t *testing.T) {
	sup, root := newSup(t, nil)
	model := nodetest.ModelDir(t)
	id := ids.New()
	asg := nodetest.Assignment(id, "fake-27b", model, 2, 1)
	sup.Apply([]wire.Assignment{asg})
	ready := waitFor(t, sup, 20*time.Second, state(id, wire.StateReady))
	if ready[0].MaxBatch != 2 {
		t.Fatalf("max_batch = %d; a keyed engine hides max_batch_size, so slots must be reported", ready[0].MaxBatch)
	}
	r, ok := sup.Get(id)
	if !ok {
		t.Fatal("ready replica not found")
	}
	base, keys := r.Endpoint()
	key := keys.Inference
	req, _ := http.NewRequest("GET", base+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var models struct{ Data []struct{ ID string } }
	_ = json.NewDecoder(resp.Body).Decode(&models)
	resp.Body.Close()
	if len(models.Data) != 1 || models.Data[0].ID != "fake-27b" {
		t.Fatalf("models %+v", models)
	}
	rel1, err1 := r.Acquire()
	rel2, err2 := r.Acquire()
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	if _, err := r.Acquire(); !errors.Is(err, engine.ErrBusy) {
		t.Fatalf("third acquire = %v, want busy", err)
	}
	rel1()
	rel1()
	rel2()
	first := pid(t, root, id)
	asg.Slots, asg.Generation = 3, 2
	sup.Apply([]wire.Assignment{asg})
	waitFor(t, sup, 5*time.Second, func(ss []wire.ReplicaStatus) bool {
		return len(ss) == 1 && ss[0].Generation == 2 && ss[0].State == wire.StateReady
	})
	if got := pid(t, root, id); got != first {
		t.Fatalf("slot change restarted the engine: pid %d -> %d", first, got)
	}
	var rels []func()
	for i := 0; i < 3; i++ {
		rel, err := r.Acquire()
		if err != nil {
			t.Fatalf("acquire %d after resize: %v", i, err)
		}
		rels = append(rels, rel)
	}
	for _, rel := range rels {
		rel()
	}
	asg.Spec.Context, asg.Generation = 4096, 3
	sup.Apply([]wire.Assignment{asg})
	waitFor(t, sup, 20*time.Second, func(ss []wire.ReplicaStatus) bool {
		return len(ss) == 1 && ss[0].State == wire.StateReady && pid(t, root, id) != first && pid(t, root, id) != 0
	})
	sup.Apply(nil)
	waitFor(t, sup, 20*time.Second, func(ss []wire.ReplicaStatus) bool { return len(ss) == 0 })
	if _, err := os.Stat(filepath.Join(root, id)); !os.IsNotExist(err) {
		t.Fatalf("replica directory left behind: %v", err)
	}
}

func TestCrashRestartsThenFailsUntilNewGeneration(t *testing.T) {
	sup, _ := newSup(t, func(c *engine.Config) { c.Env = []string{"FAKE_ENGINE_CRASH_AFTER_MS=600"} })
	model := nodetest.ModelDir(t)
	id := ids.New()
	asg := nodetest.Assignment(id, "crashy", model, 1, 1)
	sup.Apply([]wire.Assignment{asg})
	waitFor(t, sup, 20*time.Second, state(id, wire.StateReady))
	ss := waitFor(t, sup, 30*time.Second, state(id, wire.StateFailed))
	if !strings.Contains(ss[0].Error, "failed 3 times") || !strings.Contains(ss[0].Error, "fatal: simulated crash") {
		t.Fatalf("failure error %q", ss[0].Error)
	}
	time.Sleep(500 * time.Millisecond)
	if s := sup.Statuses(); s[0].State != wire.StateFailed {
		t.Fatalf("a spent generation was retried: %+v", s)
	}
	asg.Generation = 2
	sup.Apply([]wire.Assignment{asg})
	waitFor(t, sup, 20*time.Second, func(ss []wire.ReplicaStatus) bool {
		return len(ss) == 1 && ss[0].Generation == 2 && ss[0].State != wire.StateFailed
	})
}

func TestStartupTimeoutFails(t *testing.T) {
	sup, _ := newSup(t, func(c *engine.Config) {
		c.Env = []string{"FAKE_ENGINE_SILENT=1"}
		c.StartupTimeout = time.Second
		c.CrashLimit = 1
	})
	id := ids.New()
	sup.Apply([]wire.Assignment{nodetest.Assignment(id, "silent", nodetest.ModelDir(t), 1, 1)})
	ss := waitFor(t, sup, 15*time.Second, state(id, wire.StateFailed))
	if !strings.Contains(ss[0].Error, "did not announce a port") {
		t.Fatalf("error %q", ss[0].Error)
	}
}

func TestInvalidAssignmentNeverStarts(t *testing.T) {
	sup, root := newSup(t, nil)
	id := ids.New()
	a := nodetest.Assignment(id, "--api-key=x", nodetest.ModelDir(t), 1, 1)
	sup.Apply([]wire.Assignment{a, {ReplicaID: "not-a-uuid"}})
	ss := waitFor(t, sup, 5*time.Second, state(id, wire.StateFailed))
	if len(ss) != 1 || !strings.Contains(ss[0].Error, "served name") {
		t.Fatalf("statuses %+v", ss)
	}
	if pid(t, root, id) != 0 {
		t.Fatal("an invalid assignment started a process")
	}
}

func TestMissingArtifactFailsStaging(t *testing.T) {
	sup, _ := newSup(t, func(c *engine.Config) { c.CrashLimit = 1 })
	id := ids.New()
	sup.Apply([]wire.Assignment{nodetest.Assignment(id, "m", filepath.Join(t.TempDir(), "absent"), 1, 1)})
	ss := waitFor(t, sup, 5*time.Second, state(id, wire.StateFailed))
	if !strings.Contains(ss[0].Error, "stage model") {
		t.Fatalf("error %q", ss[0].Error)
	}
}

func TestRemovalDrainsInFlightFirst(t *testing.T) {
	sup, _ := newSup(t, nil)
	id := ids.New()
	sup.Apply([]wire.Assignment{nodetest.Assignment(id, "m", nodetest.ModelDir(t), 1, 1)})
	waitFor(t, sup, 20*time.Second, state(id, wire.StateReady))
	r, _ := sup.Get(id)
	rel, err := r.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	sup.Apply(nil)
	waitFor(t, sup, 5*time.Second, state(id, wire.StateDraining))
	time.Sleep(300 * time.Millisecond)
	if len(sup.Statuses()) != 1 {
		t.Fatal("replica stopped while a request was in flight")
	}
	rel()
	waitFor(t, sup, 20*time.Second, func(ss []wire.ReplicaStatus) bool { return len(ss) == 0 })
}

func TestStatsReportsMeasureThroughput(t *testing.T) {
	sup, _ := newSup(t, func(c *engine.Config) { c.Env = []string{"FAKE_ENGINE_SYNTHETIC_TPS=200"} })
	id := ids.New()
	sup.Apply([]wire.Assignment{nodetest.Assignment(id, "m", nodetest.ModelDir(t), 2, 1)})
	waitFor(t, sup, 20*time.Second, state(id, wire.StateReady))
	reports := make(chan wire.StatsReport, 16)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sup.RunMetrics(ctx, 250*time.Millisecond, func(r wire.StatsReport) {
		select {
		case reports <- r:
		default:
		}
	})
	first := <-reports
	if len(first.Replicas) != 1 || first.Replicas[0].ReplicaID != id || first.Replicas[0].TokensPerSecond != 0 {
		t.Fatalf("first report must carry the replica without a rate: %+v", first)
	}
	var got wire.ReplicaStats
	for i := 0; i < 8; i++ {
		got = (<-reports).Replicas[0]
		if got.TokensPerSecond > 0 {
			break
		}
	}
	if got.TokensPerSecond < 100 || got.TokensPerSecond > 400 || got.PromptTokensPerSecond < 400 || got.PromptTokensPerSecond > 1600 {
		t.Fatalf("rates out of range for 200 tok/s: %+v", got)
	}
	if got.Running != 0 || got.Waiting != 0 || got.KVUsage != 0 {
		t.Fatalf("idle engine gauges %+v", got)
	}
}

func TestNoReportWhenNothingIsReady(t *testing.T) {
	sup, _ := newSup(t, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()
	sent := 0
	sup.RunMetrics(ctx, 100*time.Millisecond, func(wire.StatsReport) { sent++ })
	if sent != 0 {
		t.Fatalf("sent %d reports with no ready replica", sent)
	}
}
