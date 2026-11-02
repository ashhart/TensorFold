package scim

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ratelimit"
)

// caller is the token a request authenticated with and the one connection it may provision.
type caller struct {
	TokenID      string
	ConnectionID string
}

type callerKey struct{}

func callerFrom(ctx context.Context) caller {
	c, _ := ctx.Value(callerKey{}).(caller)
	return c
}

// authenticate accepts only an unrevoked, unexpired SCIM bearer token of an enabled connection.
func (h *Handler) authenticate(r *http.Request) (caller, error) {
	scheme, raw, ok := strings.Cut(r.Header.Get("Authorization"), " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") {
		return caller{}, errUnauthorized
	}
	raw = strings.TrimSpace(raw)
	if k, ok := secret.Parse(raw); !ok || k != secret.SCIM {
		return caller{}, errUnauthorized
	}
	var c caller
	var lastUsed *time.Time
	err := h.pool.QueryRow(r.Context(), `SELECT t.id, t.connection_id, t.last_used_at FROM scim_tokens t
		JOIN idp_connections c ON c.id = t.connection_id
		WHERE t.token_hash = $1 AND t.revoked_at IS NULL AND (t.expires_at IS NULL OR t.expires_at > now()) AND c.enabled`,
		secret.Hash(raw)).Scan(&c.TokenID, &c.ConnectionID, &lastUsed)
	if db.NotFound(err) {
		return caller{}, errUnauthorized
	}
	if err != nil {
		return caller{}, err
	}
	if lastUsed == nil || time.Since(*lastUsed) > time.Minute {
		_, _ = h.pool.Exec(r.Context(), "UPDATE scim_tokens SET last_used_at = now() WHERE id = $1", c.TokenID)
	}
	return c, nil
}

// newLimiter bounds requests per token across every instance; IdPs back off on 429 with Retry-After.
func newLimiter(pool *pgxpool.Pool, key []byte, rate float64, burst int) *ratelimit.Limiter {
	if rate <= 0 {
		rate = 25
	}
	if burst <= 0 {
		burst = 200
	}
	return ratelimit.Shared(pool, key, "scim", rate, burst)
}
