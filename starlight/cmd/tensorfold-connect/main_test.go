package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/connect/fakestarlight"
	"github.com/ashhart/TensorFold/starlight/internal/connect/profile"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
)

func TestCommandsAgainstFakeStarlight(t *testing.T) {
	t.Setenv(store.BackendEnv, "file")
	t.Setenv(profile.HomeEnv, t.TempDir())
	f := fakestarlight.New(t)
	var out, errOut bytes.Buffer
	oldOut, oldErr := stdout, stderr
	stdout, stderr = &out, &errOut
	t.Cleanup(func() { stdout, stderr = oldOut, oldErr })
	run := func(args ...string) string {
		t.Helper()
		out.Reset()
		errOut.Reset()
		if err := cli.Dispatch(context.Background(), root(), args, &out, &errOut); err != nil {
			t.Fatalf("%v: %v\n%s", args, err, errOut.String())
		}
		return out.String()
	}
	if s := run("status"); !strings.Contains(s, "not configured") {
		t.Fatalf("status before login:\n%s", s)
	}
	if s := run("login", "--server", f.URL, "--device"); !strings.Contains(s, "Signed in to Example Org as ada@example.com") ||
		!strings.Contains(s, "project research") || !strings.Contains(errOut.String(), "BCDF-GHJK") {
		t.Fatalf("login:\n%s\n%s", s, errOut.String())
	}
	if s := run("models"); !strings.Contains(s, "qwen3.8-27b") {
		t.Fatalf("models:\n%s", s)
	}
	if s := run("projects"); !strings.Contains(s, "*  research") {
		t.Fatalf("projects:\n%s", s)
	}
	key := strings.TrimSpace(run("apps", "add", "notebook"))
	if !strings.HasPrefix(key, "slc_") {
		t.Fatalf("apps add printed %q", key)
	}
	if s := run("apps", "list"); !strings.Contains(s, "notebook") || strings.Contains(s, key) {
		t.Fatalf("apps list:\n%s", s)
	}
	if s := run("env", "openai", "--port", "7399", "--shell", "posix"); !strings.Contains(s, `OPENAI_BASE_URL="http://127.0.0.1:7399/v1"`) {
		t.Fatalf("env:\n%s", s)
	}
	run("apps", "remove", "notebook")
	if s := run("logout"); !strings.Contains(s, "Signed out.") {
		t.Fatalf("logout:\n%s", s)
	}
	if s := run("status"); !strings.Contains(s, "nobody") {
		t.Fatalf("status after logout:\n%s", s)
	}
}
