package fleet

import (
	"context"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// LimitKinds lists every limit kind in display order; cost kinds are in micro-dollars.
var LimitKinds = []string{"concurrency", "requests_per_minute", "tokens_per_minute", "tokens_per_day", "tokens_per_month", "cost_per_day", "cost_per_month"}

// IsCost reports whether a limit kind is denominated in money.
func IsCost(kind string) bool { return kind == "cost_per_day" || kind == "cost_per_month" }

// Limit is one row of the limits table.
type Limit struct {
	ID         string
	ScopeType  string
	ScopeID    string
	ModelID    string
	ModelName  string
	Kind       string
	Value      int64
	BoostValue *int64
	BoostUntil *time.Time
}

// Boosted reports whether a temporary boost is in force.
func (l Limit) Boosted() bool {
	return l.BoostValue != nil && l.BoostUntil != nil && l.BoostUntil.After(time.Now())
}

// Amount formats a value of this limit's kind.
func (l Limit) Amount(v int64) string {
	if IsCost(l.Kind) {
		return Dollars(v)
	}
	return Count(v)
}

// BoostAmount formats the boost value.
func (l Limit) BoostAmount() string {
	if l.BoostValue == nil {
		return ""
	}
	return l.Amount(*l.BoostValue)
}

// LimitInput sets a limit; empty ScopeID is organisation scope, empty ModelID applies to every model.
type LimitInput struct {
	ScopeType  string
	ScopeID    string
	ModelID    string
	Kind       string
	Value      int64
	BoostValue *int64
	BoostUntil *time.Time
}

func (in LimitInput) validate() error {
	if err := oneOf("limit kind", in.Kind, LimitKinds...); err != nil {
		return err
	}
	if err := oneOf("scope", in.ScopeType, "org", "project", "member", "user", "service_account", "api_key"); err != nil {
		return err
	}
	if (in.ScopeType == "org") != (in.ScopeID == "") || in.ScopeID != "" && !ids.Valid(in.ScopeID) {
		return Invalid("organisation limits have no scope id; every other scope needs one")
	}
	if in.Value < 0 || in.BoostValue != nil && *in.BoostValue < 0 {
		return Invalid("limits cannot be negative")
	}
	if (in.BoostValue == nil) != (in.BoostUntil == nil) {
		return Invalid("a boost needs both a value and an end time")
	}
	if in.BoostUntil != nil && (in.BoostUntil.Before(time.Now()) || in.BoostUntil.After(time.Now().AddDate(0, 3, 0))) {
		return Invalid("a boost must end in the future and within three months")
	}
	return nil
}

// Limits lists limits in one scope; ScopeID empty with scopeType org lists organisation limits.
func Limits(ctx context.Context, q db.Querier, scopeTypes []string, scopeID string) ([]Limit, error) {
	rows, err := q.Query(ctx, `SELECT l.id, l.scope_type, coalesce(l.scope_id::text, ''), coalesce(l.model_id::text, ''), coalesce(m.name, ''), l.kind,
		l.value, l.boost_value, l.boost_until FROM limits l LEFT JOIN models m ON m.id = l.model_id
		WHERE l.scope_type = ANY($1) AND coalesce(l.scope_id::text, '') = $2 ORDER BY l.scope_type, l.kind, m.name NULLS FIRST`, scopeTypes, scopeID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Limit
	for rows.Next() {
		var l Limit
		if err := rows.Scan(&l.ID, &l.ScopeType, &l.ScopeID, &l.ModelID, &l.ModelName, &l.Kind, &l.Value, &l.BoostValue, &l.BoostUntil); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, rows.Err()
}

// SetLimit creates or replaces the limit for its scope, model and kind.
func (t *Tx) SetLimit(ctx context.Context, in LimitInput) error {
	if err := in.validate(); err != nil {
		return err
	}
	var id string
	err := t.tx.QueryRow(ctx, `INSERT INTO limits (id, scope_type, scope_id, model_id, kind, value, boost_value, boost_until)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		ON CONFLICT (scope_type, scope_id, model_id, kind) DO UPDATE SET value = EXCLUDED.value, boost_value = EXCLUDED.boost_value,
		boost_until = EXCLUDED.boost_until, updated_at = now() RETURNING id`,
		ids.New(), in.ScopeType, optID(in.ScopeID), optID(in.ModelID), in.Kind, in.Value, in.BoostValue, in.BoostUntil).Scan(&id)
	if db.IsForeignKey(err) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	return t.record(ctx, "limit.set", "limit", id, map[string]any{"scope_type": in.ScopeType, "scope_id": in.ScopeID, "model_id": in.ModelID,
		"kind": in.Kind, "value": in.Value, "boost_value": in.BoostValue, "boost_until": in.BoostUntil})
}

// RemoveLimit deletes a limit within the expected scope.
func (t *Tx) RemoveLimit(ctx context.Context, id string, scopeTypes []string, scopeID string) error {
	var kind string
	err := t.tx.QueryRow(ctx, "DELETE FROM limits WHERE id = $1 AND scope_type = ANY($2) AND coalesce(scope_id::text, '') = $3 RETURNING kind",
		id, scopeTypes, scopeID).Scan(&kind)
	if err != nil {
		return notFound(err)
	}
	return t.record(ctx, "limit.removed", "limit", id, map[string]any{"kind": kind, "scope_id": scopeID})
}
