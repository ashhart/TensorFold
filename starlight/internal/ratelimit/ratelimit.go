// Package ratelimit limits abuse-prone endpoints per key, in memory first, then across all instances through Postgres.
package ratelimit

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"math"
	"net/netip"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// sharedTimeout bounds how long the shared check can delay a request before it fails open.
const sharedTimeout = 500 * time.Millisecond

// Limiter allows Rate events per second per key with bursts up to Burst; a shared limiter holds all instances to that.
type Limiter struct {
	mu      sync.Mutex
	rate    float64
	burst   float64
	buckets map[string]*bucket
	swept   time.Time
	now     func() time.Time
	pool    *pgxpool.Pool
	name    string
	secret  []byte
	window  time.Duration
	budget  int64
	warned  time.Time
	warn    func(msg string, args ...any)
}

type bucket struct {
	tokens float64
	last   time.Time
}

// New returns a limiter that counts on this instance only; keys idle long enough to refill are forgotten.
func New(perSecond float64, burst int) *Limiter {
	return &Limiter{rate: perSecond, burst: float64(burst), buckets: map[string]*bucket{}, now: time.Now, warn: slog.Warn}
}

// Shared returns a limiter counted per fixed window under rl:name keys, shared by every instance given the same secret.
func Shared(pool *pgxpool.Pool, secret []byte, name string, perSecond float64, burst int) *Limiter {
	l := New(perSecond, burst)
	l.pool, l.name, l.secret = pool, name, secret
	// Without a secret, a random one keeps keys irreversible, though no other instance can match them.
	if len(secret) == 0 {
		l.secret = make([]byte, 32)
		_, _ = rand.Read(l.secret)
	}
	l.window, l.budget = window(perSecond, burst)
	return l
}

// Client keys a client address: IPv6 by its /64, which one subscriber holds whole, and every unknown address as one.
func Client(a netip.Addr) string {
	a = a.Unmap()
	switch {
	case !a.IsValid():
		return "unknown"
	case a.Is6():
		p, _ := a.Prefix(64)
		return p.String()
	}
	return a.String()
}

// window is the time a full bucket takes to refill, kept between a second and an hour, and the events it allows.
func window(perSecond float64, burst int) (time.Duration, int64) {
	w := time.Duration(float64(burst) / perSecond * float64(time.Second)).Round(time.Millisecond)
	w = min(max(w, time.Second), time.Hour)
	return w, max(int64(math.Round(perSecond*w.Seconds())), 1)
}

// Allow spends one token for key and reports whether one was available.
func (l *Limiter) Allow(key string) bool {
	ok, _ := l.Take(context.Background(), key)
	return ok
}

// Take spends one token for key and, when refused, reports how long until the key may try again.
func (l *Limiter) Take(ctx context.Context, key string) (bool, time.Duration) {
	if ok, wait := l.local(key); !ok || l.pool == nil {
		return ok, wait
	}
	return l.shared(ctx, key)
}

func (l *Limiter) local(key string) (bool, time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	if now.Sub(l.swept) > time.Minute {
		full := time.Duration(l.burst / l.rate * float64(time.Second))
		for k, b := range l.buckets {
			if now.Sub(b.last) > full {
				delete(l.buckets, k)
			}
		}
		l.swept = now
	}
	b, ok := l.buckets[key]
	if !ok {
		b = &bucket{tokens: l.burst, last: now}
		l.buckets[key] = b
	}
	b.tokens = min(l.burst, b.tokens+now.Sub(b.last).Seconds()*l.rate)
	b.last = now
	if b.tokens < 1 {
		return false, time.Duration((1 - b.tokens) / l.rate * float64(time.Second))
	}
	b.tokens--
	return true, 0
}

// shared counts key in the current window; when Postgres cannot answer it allows, leaving the local bucket in charge.
func (l *Limiter) shared(ctx context.Context, key string) (bool, time.Duration) {
	now := l.now()
	start := now.Truncate(l.window)
	qctx, cancel := context.WithTimeout(ctx, sharedTimeout)
	defer cancel()
	var used int64
	err := l.pool.QueryRow(qctx, `INSERT INTO counters (key, window_start, used) VALUES ($1, $2, 1)
		ON CONFLICT (key, window_start) DO UPDATE SET used = counters.used + 1 RETURNING used`, l.key(key), start).Scan(&used)
	if err != nil {
		if ctx.Err() == nil {
			l.failed(now, err)
		}
		return true, 0
	}
	if used > l.budget {
		return false, start.Add(l.window).Sub(now)
	}
	return true, 0
}

// key is an HMAC of the caller's key under the secret, so counters never reveal client addresses, tokens or user ids.
func (l *Limiter) key(k string) string {
	mac := hmac.New(sha256.New, l.secret)
	mac.Write([]byte(l.name + "\x00" + k))
	return "rl:" + l.name + ":" + hex.EncodeToString(mac.Sum(nil)[:16])
}

func (l *Limiter) failed(now time.Time, err error) {
	l.mu.Lock()
	due := now.Sub(l.warned) >= time.Minute
	if due {
		l.warned = now
	}
	l.mu.Unlock()
	if due {
		l.warn("shared rate limit unavailable; limiting this instance only", "limiter", l.name, "err", err)
	}
}
