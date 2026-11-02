package scheduler

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
)

// housekeeper runs periodic maintenance on the leader only.
type housekeeper struct {
	pool                      *pgxpool.Pool
	hourly                    []func(context.Context) error
	reaped, pruned, partition time.Time
}

func newHousekeeper(pool *pgxpool.Pool, hourly []func(context.Context) error) *housekeeper {
	return &housekeeper{pool: pool, hourly: hourly}
}

var expiry = []string{
	"DELETE FROM auth_requests WHERE expires_at < now() - interval '1 day'",
	"DELETE FROM oauth_codes WHERE expires_at < now() - interval '1 day'",
	"DELETE FROM device_grants WHERE expires_at < now() - interval '1 day'",
	"DELETE FROM access_tokens WHERE expires_at < now() - interval '1 day'",
	"DELETE FROM refresh_tokens WHERE family_expires_at < now() - interval '30 days' OR revoked_at < now() - interval '30 days'",
	"DELETE FROM sessions WHERE expires_at < now() - interval '30 days' OR revoked_at < now() - interval '30 days'",
	"DELETE FROM response_objects WHERE expires_at < now()",
	"DELETE FROM recovery_codes WHERE expires_at < now() - interval '30 days'",
	"DELETE FROM enrollment_tokens WHERE expires_at < now() - interval '30 days'",
	"DELETE FROM node_certificates WHERE not_after < now() - interval '30 days'",
}

func (h *housekeeper) run(ctx context.Context) {
	now := time.Now()
	if now.Sub(h.reaped) >= 15*time.Second {
		_, _ = admission.Reap(ctx, h.pool)
		h.reaped = now
	}
	if now.Sub(h.pruned) >= time.Hour {
		_ = admission.Prune(ctx, h.pool)
		for _, s := range expiry {
			_, _ = h.pool.Exec(ctx, s)
		}
		for _, f := range h.hourly {
			_ = f(ctx)
		}
		_, _ = h.pool.Exec(ctx, `DELETE FROM audit_events WHERE created_at < now() - make_interval(days => (SELECT coalesce((policy->>'audit_retention_days')::int, 400)
			FROM settings WHERE id)) AND (SELECT coalesce((policy->>'audit_retention_days')::int, 400) FROM settings WHERE id) > 0`)
		h.pruned = now
	}
	if now.Sub(h.partition) >= 6*time.Hour {
		_ = admission.EnsurePartitions(ctx, h.pool, now)
		h.partition = now
	}
}
