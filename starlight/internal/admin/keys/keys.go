// Package keys issues, rotates and revokes API keys; callers record the audit event in the same transaction.
package keys

import (
	"context"
	"errors"
	"fmt"
	"net/netip"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

// Spec is a key to issue.
type Spec struct {
	Name           string
	OwnerType      string
	OwnerID        string
	ProjectID      string
	Models         []string
	Priority       string
	Classification string
	CIDRs          []string
	ExpiresAt      *time.Time
	CreatedBy      string
}

// Key is a stored key without its secret.
type Key struct {
	ID             string
	PublicID       string
	Name           string
	OwnerType      string
	OwnerID        string
	ProjectID      string
	Models         []string
	Priority       string
	Classification *string
	CIDRs          []string
	ExpiresAt      *time.Time
	LastUsedAt     *time.Time
	RevokedAt      *time.Time
	GraceUntil     *time.Time
	CreatedAt      time.Time
}

// Live reports whether the key still authenticates.
func (k Key) Live(now time.Time) bool {
	return k.RevokedAt == nil && (k.ExpiresAt == nil || k.ExpiresAt.After(now)) && (k.GraceUntil == nil || k.GraceUntil.After(now))
}

// ErrNotFound means no such key.
var ErrNotFound = errors.New("keys: not found")

func (s *Spec) validate() error {
	s.Name = strings.TrimSpace(s.Name)
	if s.Name == "" || len(s.Name) > 80 {
		return errors.New("key name is required and at most 80 characters")
	}
	if s.OwnerType != "user" && s.OwnerType != "service_account" || !ids.Valid(s.OwnerID) || !ids.Valid(s.ProjectID) {
		return errors.New("key owner and project are required")
	}
	if s.Priority == "" {
		s.Priority = "standard"
	}
	if _, err := levels.ParsePriority(s.Priority); err != nil {
		return err
	}
	if s.Classification != "" {
		if _, err := levels.ParseClass(s.Classification); err != nil {
			return err
		}
	}
	for _, c := range s.CIDRs {
		if _, err := netip.ParsePrefix(c); err != nil {
			return fmt.Errorf("allowed network %q is not CIDR notation", c)
		}
	}
	if s.ExpiresAt != nil && (s.ExpiresAt.Before(time.Now()) || s.ExpiresAt.After(time.Now().AddDate(2, 0, 0))) {
		return errors.New("expiry must be in the future and within two years")
	}
	if len(s.Models) > 64 {
		return errors.New("at most 64 models per key")
	}
	return nil
}

const cols = `id, public_id, name, owner_type, owner_id, project_id, models, priority, classification, allowed_cidrs::text[], expires_at,
	last_used_at, revoked_at, grace_until, created_at`

func scan(row pgx.Row) (Key, error) {
	var k Key
	err := row.Scan(&k.ID, &k.PublicID, &k.Name, &k.OwnerType, &k.OwnerID, &k.ProjectID, &k.Models, &k.Priority, &k.Classification, &k.CIDRs,
		&k.ExpiresAt, &k.LastUsedAt, &k.RevokedAt, &k.GraceUntil, &k.CreatedAt)
	if db.NotFound(err) {
		return k, ErrNotFound
	}
	return k, err
}

// Issue stores a new key and returns its secret, which is never shown again.
func Issue(ctx context.Context, q db.Querier, s Spec) (string, Key, error) {
	if err := s.validate(); err != nil {
		return "", Key{}, err
	}
	tok, hash := secret.Generate(secret.APIKey)
	var class *string
	if s.Classification != "" {
		class = &s.Classification
	}
	var by *string
	if ids.Valid(s.CreatedBy) {
		by = &s.CreatedBy
	}
	cidrs := s.CIDRs
	if cidrs == nil {
		cidrs = []string{}
	}
	k, err := scan(q.QueryRow(ctx, `INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id, models, priority,
		classification, allowed_cidrs, expires_at, created_by) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::cidr[],$12,$13) RETURNING `+cols,
		ids.New(), secret.PublicID(tok), hash, s.Name, s.OwnerType, s.OwnerID, s.ProjectID, s.Models, s.Priority, class, cidrs, s.ExpiresAt, by))
	return tok, k, err
}

// Get loads one key.
func Get(ctx context.Context, q db.Querier, id string) (Key, error) {
	if !ids.Valid(id) {
		return Key{}, ErrNotFound
	}
	return scan(q.QueryRow(ctx, "SELECT "+cols+" FROM api_keys WHERE id = $1", id))
}

// Revoke ends a key immediately.
func Revoke(ctx context.Context, q db.Querier, id string) error {
	tag, err := q.Exec(ctx, "UPDATE api_keys SET revoked_at = now() WHERE id = $1 AND revoked_at IS NULL", id)
	if err == nil && tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return err
}

// Rotate issues a replacement with the same scope and lets the old key work for grace, at most seven days.
func Rotate(ctx context.Context, q db.Querier, id string, grace time.Duration, by string) (string, Key, error) {
	old, err := Get(ctx, q, id)
	if err != nil {
		return "", Key{}, err
	}
	if !old.Live(time.Now()) {
		return "", Key{}, errors.New("only a live key can be rotated")
	}
	grace = min(max(grace, 0), 7*24*time.Hour)
	class := ""
	if old.Classification != nil {
		class = *old.Classification
	}
	tok, k, err := Issue(ctx, q, Spec{Name: old.Name, OwnerType: old.OwnerType, OwnerID: old.OwnerID, ProjectID: old.ProjectID, Models: old.Models,
		Priority: old.Priority, Classification: class, CIDRs: old.CIDRs, ExpiresAt: old.ExpiresAt, CreatedBy: by})
	if err != nil {
		return "", Key{}, err
	}
	if _, err := q.Exec(ctx, "UPDATE api_keys SET rotated_from = $2 WHERE id = $1", k.ID, old.ID); err != nil {
		return "", Key{}, err
	}
	_, err = q.Exec(ctx, "UPDATE api_keys SET grace_until = now() + make_interval(secs => $2) WHERE id = $1", old.ID, grace.Seconds())
	return tok, k, err
}

// Filter selects keys to list; empty fields do not filter.
type Filter struct {
	OwnerType string
	OwnerID   string
	ProjectID string
	LiveOnly  bool
}

// List returns keys newest first.
func List(ctx context.Context, q db.Querier, f Filter) ([]Key, error) {
	sql := "SELECT " + cols + " FROM api_keys WHERE ($1 = '' OR owner_type = $1) AND ($2 = '' OR owner_id::text = $2) AND ($3 = '' OR project_id::text = $3)"
	if f.LiveOnly {
		sql += " AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now()) AND (grace_until IS NULL OR grace_until > now())"
	}
	rows, err := q.Query(ctx, sql+" ORDER BY created_at DESC LIMIT 500", f.OwnerType, f.OwnerID, f.ProjectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Key
	for rows.Next() {
		k, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, k)
	}
	return out, rows.Err()
}
