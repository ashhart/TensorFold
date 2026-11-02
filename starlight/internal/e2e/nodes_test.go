package e2e_test

import (
	"testing"
	"time"
)

// setNode mirrors the console and CLI node controls: status change, desired-state bump and a sync wake-up.
func (f *fleet) setNode(status string) {
	f.t.Helper()
	var id string
	if err := f.pool.QueryRow(f.ctx, "UPDATE nodes SET status = $1, sync_generation = sync_generation + 1 WHERE name = 'studio-1' RETURNING id", status).
		Scan(&id); err != nil {
		f.t.Fatal(err)
	}
	switch status {
	case "draining":
		exec(f.t, f.pool, "UPDATE replicas SET desired = 'stopped', generation = generation + 1 WHERE node_id = $1", id)
	case "revoked":
		exec(f.t, f.pool, "UPDATE node_certificates SET revoked_at = now() WHERE node_id = $1", id)
		exec(f.t, f.pool, "UPDATE replicas SET desired = 'stopped', state = 'lost' WHERE node_id = $1", id)
	}
	exec(f.t, f.pool, "SELECT pg_notify('starlight_node', $1)", id)
}

func (f *fleet) waitReplica(want string, timeout time.Duration) {
	f.t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		var n int
		_ = f.pool.QueryRow(f.ctx, "SELECT count(*) FROM replicas WHERE state = $1 AND desired = 'running'", want).Scan(&n)
		if n > 0 || want == "none" && f.liveReplicas() == 0 {
			return
		}
		time.Sleep(150 * time.Millisecond)
	}
	f.t.Fatalf("no replica reached %q within %s", want, timeout)
}

func (f *fleet) liveReplicas() int {
	var n int
	_ = f.pool.QueryRow(f.ctx, "SELECT count(*) FROM replicas WHERE desired = 'running' AND state NOT IN ('stopped', 'lost', 'failed')").Scan(&n)
	return n
}

func TestNodeOnlineControls(t *testing.T) {
	f := startFleet(t)
	chat := `{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}`
	f.setNode("draining")
	f.waitReplica("none", 20*time.Second)
	var stopped int
	deadline := time.Now().Add(20 * time.Second)
	for stopped == 0 && time.Now().Before(deadline) {
		_ = f.pool.QueryRow(f.ctx, "SELECT count(*) FROM replicas WHERE desired = 'stopped' AND state = 'stopped'").Scan(&stopped)
		time.Sleep(150 * time.Millisecond)
	}
	if stopped == 0 {
		t.Fatal("the drained node never reported its engine stopped")
	}
	if resp, body := f.call("POST", "/v1/chat/completions", f.key, chat); resp.StatusCode != 503 {
		t.Fatalf("request with every node drained %d %s", resp.StatusCode, body)
	}
	f.setNode("active")
	f.waitReplica("ready", 60*time.Second)
	if resp, body := f.call("POST", "/v1/chat/completions", f.key, chat); resp.StatusCode != 200 {
		t.Fatalf("request after bringing the node back %d %s", resp.StatusCode, body)
	}
	f.setNode("revoked")
	time.Sleep(time.Second)
	if resp, _ := f.call("POST", "/v1/chat/completions", f.key, chat); resp.StatusCode != 503 {
		t.Fatalf("request after revoking the only node answered %d", resp.StatusCode)
	}
}
