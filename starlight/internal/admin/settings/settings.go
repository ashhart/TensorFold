// Package settings edits installation-wide settings and reads the audit trail for the console.
package settings

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Settings is the installation's settings row.
type Settings struct {
	InstallationID     string
	TrustDomain        string
	OrgName            string
	Offline            bool
	AuditRetentionDays int
	UpdatedAt          time.Time
}

// DefaultRetentionDays applies when the policy does not set a retention.
const DefaultRetentionDays = 400

// Get reads the settings.
func Get(ctx context.Context, q db.Querier) (Settings, error) {
	var s Settings
	var policy []byte
	err := q.QueryRow(ctx, "SELECT installation_id, trust_domain, org_name, offline, policy, updated_at FROM settings WHERE id").
		Scan(&s.InstallationID, &s.TrustDomain, &s.OrgName, &s.Offline, &policy, &s.UpdatedAt)
	if err != nil {
		return s, err
	}
	var p struct {
		AuditRetentionDays *int `json:"audit_retention_days"`
	}
	_ = json.Unmarshal(policy, &p)
	s.AuditRetentionDays = DefaultRetentionDays
	if p.AuditRetentionDays != nil {
		s.AuditRetentionDays = *p.AuditRetentionDays
	}
	return s, nil
}

// Change lists the settings to update; nil fields stay as they are.
type Change struct {
	OrgName            *string
	Offline            *bool
	AuditRetentionDays *int
}

// ErrInvalid explains a setting that cannot be stored.
type ErrInvalid string

func (e ErrInvalid) Error() string { return string(e) }

func (c Change) validate() error {
	if c.OrgName != nil {
		n := strings.TrimSpace(*c.OrgName)
		if n == "" || len(n) > 120 {
			return ErrInvalid("the organisation name is required and at most 120 characters")
		}
		*c.OrgName = n
	}
	if d := c.AuditRetentionDays; d != nil && *d != 0 && (*d < 30 || *d > 3650) {
		return ErrInvalid("keep audit events for 30 to 3650 days, or 0 to keep them forever")
	}
	return nil
}

// Apply validates and writes a change inside tx and returns the prior settings and the changes; it does not audit.
func Apply(ctx context.Context, tx pgx.Tx, c Change) (Settings, map[string]any, error) {
	if err := c.validate(); err != nil {
		return Settings{}, nil, err
	}
	before, err := Get(ctx, tx)
	if err != nil {
		return Settings{}, nil, err
	}
	changes := map[string]any{}
	set := func(key, sql string, from, to any) {
		if err == nil && from != to {
			changes[key] = map[string]any{"from": from, "to": to}
			_, err = tx.Exec(ctx, sql, to)
		}
	}
	if c.OrgName != nil {
		set("org_name", "UPDATE settings SET org_name = $1, updated_at = now() WHERE id", before.OrgName, *c.OrgName)
	}
	if c.Offline != nil {
		set("offline", "UPDATE settings SET offline = $1, updated_at = now() WHERE id", before.Offline, *c.Offline)
	}
	if c.AuditRetentionDays != nil {
		set("audit_retention_days", `UPDATE settings SET policy = jsonb_set(policy, '{audit_retention_days}', to_jsonb($1::int)), updated_at = now()
			WHERE id`, before.AuditRetentionDays, *c.AuditRetentionDays)
	}
	return before, changes, err
}

// Action names the audit event for a set of changes; going offline or online is its own event.
func Action(changes map[string]any) string {
	if _, ok := changes["offline"]; ok {
		return "settings.offline"
	}
	return "settings.updated"
}

// Update applies a change and records what it replaced in the same transaction.
func Update(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, c Change) (Settings, error) {
	var out Settings
	err := db.Tx(ctx, pool, func(tx pgx.Tx) error {
		before, changes, err := Apply(ctx, tx, c)
		if err != nil || len(changes) == 0 {
			out = before
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID, ev.Details = Action(changes), "settings", before.InstallationID, changes
		if err := audit.Append(ctx, tx, ev); err != nil {
			return err
		}
		out, err = Get(ctx, tx)
		return err
	})
	return out, err
}
