package engine_test

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

func TestEngineUpgradedInPlaceStartsWithoutAgentRestart(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in engine is a shell script")
	}
	dir := t.TempDir()
	version, wrapper := filepath.Join(dir, "version"), filepath.Join(dir, "tensorfold")
	if err := os.WriteFile(version, []byte("tensorfold 1.1.6\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\nif [ \"$1\" = --version ]; then cat '" + version + "'; exit 0; fi\nexec '" + fakeengine.Binary(t) + "' \"$@\"\n"
	if err := os.WriteFile(wrapper, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	sup, _ := newSup(t, func(c *engine.Config) { c.Engine, c.CrashLimit = wrapper, 1000 })
	id := ids.New()
	sup.Apply([]wire.Assignment{nodetest.Assignment(id, "upgraded", nodetest.ModelDir(t), 1, 1)})
	waitFor(t, sup, 10*time.Second, func(ss []wire.ReplicaStatus) bool {
		return len(ss) == 1 && strings.Contains(ss[0].Error, "older than 1.2.0")
	})
	if err := os.WriteFile(version, []byte("tensorfold 1.2.0\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	waitFor(t, sup, 20*time.Second, state(id, wire.StateReady))
}
