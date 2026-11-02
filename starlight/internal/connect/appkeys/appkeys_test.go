package appkeys

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestAddLookupReplaceRemove(t *testing.T) {
	path := filepath.Join(t.TempDir(), "apps.json")
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	key, err := r.Add("claude-code")
	if err != nil || !strings.HasPrefix(key, "slc_") {
		t.Fatalf("add = %q, %v", key, err)
	}
	if _, err := r.Add("claude-code"); !errors.Is(err, ErrExists) {
		t.Fatalf("duplicate add = %v", err)
	}
	if a, ok := r.Lookup(key); !ok || a.Name != "claude-code" {
		t.Fatalf("lookup = %+v, %v", a, ok)
	}
	data, _ := os.ReadFile(path)
	if strings.Contains(string(data), key) || strings.Contains(string(data), key[20:]) {
		t.Fatal("the key itself was written to disk")
	}
	if runtime.GOOS != "windows" {
		if fi, _ := os.Stat(path); fi.Mode().Perm() != 0o600 {
			t.Fatalf("apps file mode %v", fi.Mode().Perm())
		}
	}
	rotated, err := r.Replace("claude-code")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Lookup(key); ok {
		t.Fatal("replaced key still works")
	}
	if _, ok := r.Lookup(rotated); !ok {
		t.Fatal("new key rejected")
	}
	if err := r.Remove("claude-code"); err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Lookup(rotated); ok {
		t.Fatal("removed app's key still works")
	}
	if err := r.Remove("claude-code"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second remove = %v", err)
	}
}

func TestReloadSeesOtherProcess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "apps.json")
	bridge, _ := Open(path)
	cli, _ := Open(path)
	time.Sleep(10 * time.Millisecond)
	key, err := cli.Add("codex")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := bridge.Lookup(key); !ok {
		t.Fatal("a running bridge did not see a key added by another process")
	}
	if err := cli.Remove("codex"); err != nil {
		t.Fatal(err)
	}
	if _, ok := bridge.Lookup(key); ok {
		t.Fatal("a running bridge kept a removed key")
	}
}

func TestRejectsBadNamesAndKeys(t *testing.T) {
	r, _ := Open(filepath.Join(t.TempDir(), "apps.json"))
	for _, n := range []string{"", "Upper", "-x", "a b", strings.Repeat("a", 65)} {
		if _, err := r.Add(n); !errors.Is(err, ErrName) {
			t.Errorf("add %q = %v", n, err)
		}
	}
	for _, k := range []string{"", "slc_x", "sk-1234", "slr_" + strings.Repeat("a", 64)} {
		if _, ok := r.Lookup(k); ok {
			t.Errorf("accepted key %q", k)
		}
	}
}
