package fakeengine

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

const pkg = "github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine/cmd/fake-tensorfold"

var (
	buildOnce sync.Once
	binPath   string
	binDir    string
	buildErr  error
)

// BuildBinary compiles the fake engine once per test process and returns its path.
func BuildBinary() (string, error) {
	buildOnce.Do(func() {
		goBin, err := exec.LookPath("go")
		if err != nil {
			buildErr = fmt.Errorf("go toolchain not on PATH: %w", err)
			return
		}
		binDir, err = os.MkdirTemp("", "starlight-fakeengine-")
		if err != nil {
			buildErr = err
			return
		}
		name := "fake-tensorfold"
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		out := filepath.Join(binDir, name)
		cmd := exec.Command(goBin, "build", "-o", out, pkg)
		if b, err := cmd.CombinedOutput(); err != nil {
			buildErr = fmt.Errorf("build fake engine: %v: %s", err, b)
			return
		}
		binPath = out
	})
	return binPath, buildErr
}

// Binary returns the fake engine's path or fails the test.
func Binary(t testing.TB) string {
	t.Helper()
	p, err := BuildBinary()
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// RemoveBinary deletes the compiled fake engine; call it from TestMain after m.Run.
func RemoveBinary() {
	if binDir != "" {
		_ = os.RemoveAll(binDir)
	}
}
