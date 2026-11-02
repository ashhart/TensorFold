package syncer_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/node/artifacts"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/enroll"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
	"github.com/ashhart/TensorFold/starlight/internal/node/syncer"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

func TestMain(m *testing.M) {
	code := m.Run()
	fakeengine.RemoveBinary()
	os.Exit(code)
}

func TestLocalChangeInterruptsLongPollAndKeysPersist(t *testing.T) {
	ctl := nodetest.NewControl(t)
	d, _ := state.Open(filepath.Join(t.TempDir(), "state"))
	id, _, err := enroll.Run(context.Background(), d, enroll.Options{Server: ctl.URL(), Token: ctl.Token, CAPin: ctl.CAPin()})
	if err != nil {
		t.Fatal(err)
	}
	creds, _ := state.LoadCreds(d)
	client, err := enroll.ControlClient(d, id, creds, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	bin := fakeengine.Binary(t)
	stager := &artifacts.Stager{Engine: bin, ModelsDir: t.TempDir()}
	sup, err := engine.New(engine.Config{Engine: bin, Root: t.TempDir(), Stager: stager})
	if err != nil {
		t.Fatal(err)
	}
	defer sup.Shutdown(context.Background())
	_ = d.SavePermitKeys(nil)
	verifier := permit.NewVerifier(nil)
	changes := make(chan struct{}, 1)
	s := &syncer.Syncer{Client: client, Sup: sup, Verifier: verifier, Dir: d, Identity: id, Creds: creds, Stager: stager,
		Facts: func(context.Context) wire.Facts { return wire.Facts{Hostname: "h"} }, Address: "node.example:7443", Changes: changes,
		Debounce: 10 * time.Millisecond}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = s.Run(ctx); close(done) }()
	ctl.WaitSync(t, 10*time.Second, func(r wire.SyncRequest) bool { return r.Generation == 1 })
	n := len(ctl.Syncs())
	time.Sleep(100 * time.Millisecond)
	changes <- struct{}{}
	start := time.Now()
	for len(ctl.Syncs()) == n && time.Since(start) < time.Second {
		time.Sleep(10 * time.Millisecond)
	}
	if len(ctl.Syncs()) == n {
		t.Fatal("a local change did not interrupt the held long-poll")
	}
	keys, err := d.PermitKeys()
	if err != nil || keys[ctl.Signer.KID] == nil {
		t.Fatalf("permit keys not persisted: %v %v", keys, err)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("syncer did not stop")
	}
}
