package ratelimit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/netip"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestInstancesShareOneBudget(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	at := time.Date(2026, 11, 2, 9, 0, 1, 0, time.UTC)
	secret := []byte("every instance derives this")
	instances := []*Limiter{Shared(pool, secret, "test", 0.5, 10), Shared(pool, secret, "test", 0.5, 10), Shared(pool, secret, "test", 0.5, 10)}
	for _, l := range instances {
		l.now = func() time.Time { return at }
	}
	var allowed atomic.Int64
	var wg sync.WaitGroup
	for _, l := range instances[:2] {
		for range 10 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if ok, _ := l.Take(ctx, "192.0.2.10"); ok {
					allowed.Add(1)
				}
			}()
		}
	}
	wg.Wait()
	if n := allowed.Load(); n != 10 {
		t.Fatalf("two instances allowed %d together, want the budget of 10", n)
	}
	if ok, wait := instances[2].Take(ctx, "192.0.2.10"); ok || wait != 19*time.Second {
		t.Fatalf("third instance with a full bucket: allowed %v, retry after %v", ok, wait)
	}
	if !instances[2].Allow("192.0.2.11") {
		t.Fatal("another address shares the spent budget")
	}
	at = at.Add(20 * time.Second)
	if !instances[2].Allow("192.0.2.10") {
		t.Fatal("the next window still refuses")
	}
	rows, err := pool.Query(ctx, "SELECT key FROM counters")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			t.Fatal(err)
		}
		if n++; !strings.HasPrefix(key, "rl:test:") || strings.Contains(key, "192.0.2") {
			t.Fatalf("counter key %q", key)
		}
	}
	if n != 3 {
		t.Fatalf("%d counter rows, want two windows for one address and one for the other", n)
	}
}

func TestLocalBucketLimitsWithoutDatabase(t *testing.T) {
	at := time.Date(2026, 11, 2, 9, 0, 0, 0, time.UTC)
	clock := func() time.Time { return at }
	local := New(1, 3)
	local.now = clock
	for i := range 3 {
		if !local.Allow("192.0.2.20") {
			t.Fatalf("request %d refused within the burst", i+1)
		}
	}
	if ok, wait := local.Take(context.Background(), "192.0.2.20"); ok || wait != time.Second {
		t.Fatalf("burst exceeded: allowed %v, retry after %v", ok, wait)
	}
	pool, err := pgxpool.New(context.Background(), "postgres://starlight@127.0.0.1:1/none?sslmode=disable&connect_timeout=1")
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	down := Shared(pool, nil, "test", 1, 3)
	down.now = clock
	warnings := 0
	down.warn = func(string, ...any) { warnings++ }
	for i := range 3 {
		if !down.Allow("192.0.2.20") {
			t.Fatalf("request %d refused while Postgres is down", i+1)
		}
	}
	if down.Allow("192.0.2.20") {
		t.Fatal("the local bucket stopped limiting while Postgres is down")
	}
	at = at.Add(30 * time.Second)
	for range 3 {
		down.Allow("192.0.2.20")
	}
	if warnings != 1 {
		t.Fatalf("%d warnings within a minute, want 1", warnings)
	}
	at = at.Add(31 * time.Second)
	down.Allow("192.0.2.20")
	if warnings != 2 {
		t.Fatalf("%d warnings after a minute, want 2", warnings)
	}
}

func TestWindowsMatchBuckets(t *testing.T) {
	for _, c := range []struct {
		rate   float64
		burst  int
		window time.Duration
		budget int64
	}{
		{25, 200, 8 * time.Second, 200},
		{5, 30, 6 * time.Second, 30},
		{0.2, 10, 50 * time.Second, 10},
		{30.0 / 60, 30, time.Minute, 30},
		{2.0 / 60, 2, time.Minute, 2},
		{1000, 10, time.Second, 1000},
		{0.0001, 10, time.Hour, 1},
	} {
		if w, b := window(c.rate, c.burst); w != c.window || b != c.budget {
			t.Errorf("window(%v, %d) = %v, %d; want %v, %d", c.rate, c.burst, w, b, c.window, c.budget)
		}
	}
}

func TestCounterKeysCannotBeReversedWithoutTheSecret(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	at := time.Date(2026, 11, 2, 9, 0, 1, 0, time.UTC)
	ours, theirs := Shared(pool, []byte("this installation"), "login", 0.5, 1), Shared(pool, []byte("another installation"), "login", 0.5, 1)
	for _, l := range []*Limiter{ours, theirs} {
		l.now = func() time.Time { return at }
		if ok, _ := l.Take(ctx, "203.0.113.77"); !ok {
			t.Fatal("first attempt refused")
		}
	}
	sum := sha256.Sum256([]byte("login\x00203.0.113.77"))
	unsalted := "rl:login:" + hex.EncodeToString(sum[:16])
	rows, err := pool.Query(ctx, "SELECT key FROM counters")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	seen := map[string]bool{}
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			t.Fatal(err)
		}
		if key == unsalted || !strings.HasPrefix(key, "rl:login:") || len(key) != len("rl:login:")+32 {
			t.Fatalf("counter key %q can be recomputed from the address alone", key)
		}
		seen[key] = true
	}
	if len(seen) != 2 {
		t.Fatalf("%d distinct keys for one address under two secrets, want 2", len(seen))
	}
}

func TestClientKeysIPv6ByItsSlash64(t *testing.T) {
	for in, want := range map[string]string{
		"203.0.113.7":                      "203.0.113.7",
		"::ffff:203.0.113.7":               "203.0.113.7",
		"2001:db8:1:2::1":                  "2001:db8:1:2::/64",
		"2001:db8:1:2:ffff:ffff:ffff:fffe": "2001:db8:1:2::/64",
		"2001:db8:1:3::1":                  "2001:db8:1:3::/64",
		"fe80::1%eth0":                     "fe80::/64",
	} {
		if got := Client(netip.MustParseAddr(in)); got != want {
			t.Errorf("Client(%s) = %s, want %s", in, got, want)
		}
	}
	if Client(netip.Addr{}) != "unknown" || Client(netip.Addr{}) == Client(netip.MustParseAddr("10.0.0.5")) {
		t.Fatal("an unknown client address shares a real address's bucket")
	}
	at := time.Date(2026, 11, 2, 9, 0, 0, 0, time.UTC)
	l := New(0.1, 2)
	l.now = func() time.Time { return at }
	for i, a := range []string{"2001:db8:1:2::1", "2001:db8:1:2::2"} {
		if !l.Allow(Client(netip.MustParseAddr(a))) {
			t.Fatalf("attempt %d within the burst refused", i+1)
		}
	}
	if l.Allow(Client(netip.MustParseAddr("2001:db8:1:2:abcd::9"))) {
		t.Fatal("a third address in the same /64 got a fresh budget")
	}
	if !l.Allow(Client(netip.MustParseAddr("2001:db8:1:3::1"))) {
		t.Fatal("the neighbouring /64 shares the spent budget")
	}
}
