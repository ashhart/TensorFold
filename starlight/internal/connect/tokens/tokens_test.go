package tokens

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fakestarlight"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
)

func manager(t *testing.T, f *fakestarlight.Starlight) (*Manager, store.Store) {
	t.Helper()
	dir := t.TempDir()
	st := store.NewFile(dir)
	return New(st, filepath.Join(dir, "refresh.lock"), http.DefaultClient, f.Doc()), st
}

func TestRefreshIsSingleFlight(t *testing.T) {
	f := fakestarlight.New(t)
	m, _ := manager(t, f)
	if err := m.Save(f.Issue()); err != nil {
		t.Fatal(err)
	}
	stale, _ := m.Access(context.Background())
	var wg sync.WaitGroup
	results := make([]string, 20)
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			tok, err := m.Refresh(context.Background(), stale)
			if err != nil {
				t.Error(err)
			}
			results[i] = tok
		}(i)
	}
	wg.Wait()
	if n := f.RefreshCalls.Load(); n != 1 {
		t.Fatalf("%d refresh calls, want 1", n)
	}
	for _, r := range results {
		if r != results[0] || r == stale {
			t.Fatalf("callers got different tokens: %v", results)
		}
	}
}

func TestRotationPersistsNewRefreshToken(t *testing.T) {
	f := fakestarlight.New(t)
	m, st := manager(t, f)
	first := f.Issue()
	m.Save(first)
	tok1, err := m.Refresh(context.Background(), first.AccessToken)
	if err != nil {
		t.Fatal(err)
	}
	var s Set
	data, _ := st.Get(Key)
	json.Unmarshal(data, &s)
	if s.RefreshToken == first.RefreshToken || s.AccessToken != tok1 {
		t.Fatal("rotated refresh token was not stored")
	}
	if _, err := m.Refresh(context.Background(), tok1); err != nil {
		t.Fatalf("second refresh with the rotated token: %v", err)
	}
}

func TestInvalidGrantSignsOut(t *testing.T) {
	f := fakestarlight.New(t)
	m, st := manager(t, f)
	tr := f.Issue()
	m.Save(tr)
	other, _ := manager(t, f)
	other.Save(tr)
	if _, err := other.Refresh(context.Background(), tr.AccessToken); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Refresh(context.Background(), tr.AccessToken); !errors.Is(err, ErrSignedOut) {
		t.Fatalf("reused refresh token = %v", err)
	}
	if _, err := st.Get(Key); !errors.Is(err, store.ErrNotFound) {
		t.Fatal("tokens kept after invalid_grant")
	}
	if _, err := m.Access(context.Background()); !errors.Is(err, ErrSignedOut) {
		t.Fatalf("access after sign-out = %v", err)
	}
}

func TestRefreshUsesAnotherProcessesResult(t *testing.T) {
	f := fakestarlight.New(t)
	dir := t.TempDir()
	st := store.NewFile(dir)
	lock := filepath.Join(dir, "refresh.lock")
	a := New(st, lock, http.DefaultClient, f.Doc())
	b := New(st, lock, http.DefaultClient, f.Doc())
	tr := f.Issue()
	a.Save(tr)
	if _, err := b.Current(); err != nil {
		t.Fatal(err)
	}
	newTok, err := a.Refresh(context.Background(), tr.AccessToken)
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.Refresh(context.Background(), tr.AccessToken)
	if err != nil || got != newTok || f.RefreshCalls.Load() != 1 {
		t.Fatalf("second process refreshed again: %q vs %q, %d calls, %v", got, newTok, f.RefreshCalls.Load(), err)
	}
}

func TestAccessRefreshesNearExpiry(t *testing.T) {
	f := fakestarlight.New(t)
	m, _ := manager(t, f)
	tr := f.Issue()
	tr.ExpiresIn = 30
	m.Save(tr)
	tok, err := m.Access(context.Background())
	if err != nil || tok == tr.AccessToken || f.RefreshCalls.Load() != 1 {
		t.Fatalf("near-expiry access = %q, %v, %d calls", tok, err, f.RefreshCalls.Load())
	}
	m.now = func() time.Time { return time.Now().Add(-time.Hour) }
	if tok2, _ := m.Access(context.Background()); tok2 != tok {
		t.Fatal("fresh token was refreshed again")
	}
}

func TestSignOutRevokes(t *testing.T) {
	f := fakestarlight.New(t)
	m, _ := manager(t, f)
	tr := f.Issue()
	m.Save(tr)
	if err := m.SignOut(context.Background()); err != nil {
		t.Fatal(err)
	}
	other, _ := manager(t, f)
	other.Save(tr)
	if _, err := other.Refresh(context.Background(), tr.AccessToken); !errors.Is(err, ErrSignedOut) {
		t.Fatalf("revoked refresh token still works: %v", err)
	}
}
