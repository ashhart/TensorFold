package fsutil

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestWriteAtomicPermissions(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "connect")
	path := filepath.Join(dir, "secret.json")
	if err := WriteAtomic(path, []byte("one"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := WriteAtomic(path, []byte("two"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != "two" {
		t.Fatalf("read = %q, %v", got, err)
	}
	if runtime.GOOS == "windows" {
		return
	}
	fi, _ := os.Stat(path)
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("file mode %v", fi.Mode().Perm())
	}
	di, _ := os.Stat(dir)
	if di.Mode().Perm() != 0o700 {
		t.Fatalf("dir mode %v", di.Mode().Perm())
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("temporary files left behind: %v", entries)
	}
}

func TestLockIsExclusive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.lock")
	var mu sync.Mutex
	inside, maxInside := 0, 0
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			unlock, err := Lock(path)
			if err != nil {
				t.Error(err)
				return
			}
			mu.Lock()
			inside++
			maxInside = max(maxInside, inside)
			mu.Unlock()
			time.Sleep(5 * time.Millisecond)
			mu.Lock()
			inside--
			mu.Unlock()
			unlock()
		}()
	}
	wg.Wait()
	if maxInside != 1 {
		t.Fatalf("%d holders at once", maxInside)
	}
}
