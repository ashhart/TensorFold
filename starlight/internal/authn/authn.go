// Package authn turns a bearer credential into a Caller with its grants and constraints, with a short cache.
package authn

import (
	"context"
	"crypto/sha256"
	"errors"
	"net/netip"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/principal"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

// Pin limits a task to one deployment revision of a model.
type Pin struct {
	Deployment string `json:"deployment"`
	Revision   int    `json:"revision"`
}

// Task is the delegated scope carried by a task token.
type Task struct {
	ID              string
	RootID          string
	Depth           int
	MaxDepth        int
	MaxConcurrency  int
	TreeConcurrency int
	LocalOnly       bool
	Pins            map[string][]Pin
	BudgetTokens    *int64
	ExpiresAt       time.Time
}

// Caller is an authenticated principal with everything admission needs about its credential.
type Caller struct {
	P           *principal.Principal
	Grants      authz.Grants
	Models      []string
	Priority    levels.Priority
	MaxPriority levels.Priority
	ClassFloor  levels.Class
	Task        *Task
	Family      string
	KeyExpires  *time.Time
	expires     time.Time
}

// AllowsModel reports whether the credential's own model list admits name.
func (c *Caller) AllowsModel(name string) bool {
	if c.Models == nil {
		return true
	}
	for _, m := range c.Models {
		if m == name {
			return true
		}
	}
	return false
}

// JWTResolver maps an IdP-issued bearer JWT to a Starlight user id.
type JWTResolver func(ctx context.Context, raw string) (userID string, err error)

// Authenticator resolves credentials against Postgres; Timeout bounds each resolution so an outage is not waited on.
type Authenticator struct {
	Pool    *pgxpool.Pool
	JWT     JWTResolver
	TTL     time.Duration
	Timeout time.Duration
	mu      sync.Mutex
	cache   map[[32]byte]*Caller
	swept   time.Time
}

// Errors returned to the gateway: ErrInvalid maps to 401 and ErrProject to 403; database errors pass through.
var (
	ErrInvalid = errors.New("authn: invalid credentials")
	ErrProject = errors.New("authn: choose a project with X-Starlight-Project")
)

// Authenticate resolves token for a request from ip acting in the optional project hint.
func (a *Authenticator) Authenticate(ctx context.Context, token, projectHint string, ip netip.Addr) (*Caller, error) {
	if token == "" {
		return nil, ErrInvalid
	}
	key := sha256.Sum256([]byte(token + "\x00" + projectHint + "\x00" + ip.String()))
	if c := a.cached(key); c != nil {
		return c, nil
	}
	ctx, cancel := context.WithTimeout(ctx, a.timeout())
	defer cancel()
	c, err := a.resolve(ctx, token, projectHint, ip)
	if err != nil {
		return nil, err
	}
	a.store(key, c)
	return c, nil
}

func (a *Authenticator) resolve(ctx context.Context, token, hint string, ip netip.Addr) (*Caller, error) {
	if k, ok := secret.Parse(token); ok {
		switch k {
		case secret.APIKey:
			return a.apiKey(ctx, token, ip)
		case secret.Access:
			return a.access(ctx, token, hint)
		case secret.Task:
			return a.task(ctx, token, ip)
		}
		return nil, ErrInvalid
	}
	if a.JWT != nil && strings.Count(token, ".") == 2 && strings.HasPrefix(token, "eyJ") {
		userID, err := a.JWT(ctx, token)
		if err != nil {
			// A database outage or a lapsed deadline says nothing about the token; any other failure rejects it.
			if db.Unavailable(err) || ctx.Err() != nil {
				return nil, err
			}
			return nil, ErrInvalid
		}
		c, err := a.user(ctx, userID, hint, principal.ViaIDPJWT)
		if err != nil {
			return nil, err
		}
		c.expires = time.Now().Add(a.ttl())
		return c, nil
	}
	return nil, ErrInvalid
}

func (a *Authenticator) ttl() time.Duration {
	if a.TTL <= 0 {
		return 15 * time.Second
	}
	return a.TTL
}

func (a *Authenticator) timeout() time.Duration {
	if a.Timeout <= 0 {
		return db.CallTimeout
	}
	return a.Timeout
}

func (a *Authenticator) cached(key [32]byte) *Caller {
	a.mu.Lock()
	defer a.mu.Unlock()
	c := a.cache[key]
	if c == nil || time.Now().After(c.expires) {
		return nil
	}
	return c
}

func (a *Authenticator) store(key [32]byte, c *Caller) {
	limit := time.Now().Add(a.ttl())
	if c.expires.IsZero() || c.expires.After(limit) {
		c.expires = limit
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cache == nil {
		a.cache = map[[32]byte]*Caller{}
	}
	if time.Since(a.swept) > time.Minute || len(a.cache) > 100000 {
		now := time.Now()
		for k, v := range a.cache {
			if now.After(v.expires) {
				delete(a.cache, k)
			}
		}
		a.swept = now
	}
	a.cache[key] = c
}

// Forget drops every cached credential, used after revocations made by this instance.
func (a *Authenticator) Forget() {
	a.mu.Lock()
	a.cache = nil
	a.mu.Unlock()
}
