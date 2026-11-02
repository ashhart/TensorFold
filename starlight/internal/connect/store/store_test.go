package store

import (
	"bytes"
	"errors"
	"os"
	"runtime"
	"sync"
	"testing"
)

func exercise(t *testing.T, s Store, key string) {
	t.Helper()
	if _, err := s.Get(key); !errors.Is(err, ErrNotFound) {
		t.Fatalf("empty get = %v", err)
	}
	if err := s.Set(key, []byte("first")); err != nil {
		t.Fatal(err)
	}
	if err := s.Set(key, []byte("second")); err != nil {
		t.Fatal(err)
	}
	got, err := s.Get(key)
	if err != nil || !bytes.Equal(got, []byte("second")) {
		t.Fatalf("get = %q, %v", got, err)
	}
	if err := s.Delete(key); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(key); err != nil {
		t.Fatalf("second delete = %v", err)
	}
	if _, err := s.Get(key); !errors.Is(err, ErrNotFound) {
		t.Fatalf("get after delete = %v", err)
	}
}

func TestFileStore(t *testing.T) {
	dir := t.TempDir()
	f := NewFile(dir)
	exercise(t, f, "tokens")
	if err := f.Set("tokens", []byte("x")); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" {
		fi, err := os.Stat(dir + "/secrets.json")
		if err != nil || fi.Mode().Perm() != 0o600 {
			t.Fatalf("secrets file mode %v, %v", fi.Mode().Perm(), err)
		}
	}
}

func TestFileStoreConcurrentWriters(t *testing.T) {
	dir := t.TempDir()
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := NewFile(dir).Set(string(rune('a'+i)), []byte{byte(i)}); err != nil {
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()
	f := NewFile(dir)
	for i := 0; i < 16; i++ {
		if v, err := f.Get(string(rune('a' + i))); err != nil || v[0] != byte(i) {
			t.Fatalf("key %d lost: %v %v", i, v, err)
		}
	}
}

func TestBackendOverride(t *testing.T) {
	t.Setenv(BackendEnv, "file")
	if _, ok := Open(t.TempDir()).(*File); !ok {
		t.Fatal("file backend not honoured")
	}
}

func TestPlatformStore(t *testing.T) {
	if os.Getenv("TENSORFOLD_CONNECT_PLATFORM_STORE_TEST") != "1" {
		t.Skip("set TENSORFOLD_CONNECT_PLATFORM_STORE_TEST=1 to exercise the real credential store")
	}
	s, err := openPlatform()
	if err != nil {
		t.Skip(err)
	}
	exercise(t, s, "test-entry")
}
