package admission

import (
	"context"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Subject names every scope one request counts against.
type Subject struct {
	ProjectID        string
	UserID           string
	ServiceAccountID string
	KeyID            string
	ModelID          string
	TaskID           string
	TaskRootID       string
	TaskConcurrency  int
	TreeConcurrency  int
	TaskBudget       *int64
}

// Limit is one row of the limits table with any active boost applied.
type Limit struct {
	ScopeType string
	ScopeID   string
	ModelID   string
	Kind      string
	Value     int64
}

// LoadLimits reads the limits that apply to s, using a boost while it lasts.
func LoadLimits(ctx context.Context, q db.Querier, s Subject) ([]Limit, error) {
	rows, err := q.Query(ctx, `SELECT scope_type, coalesce(scope_id::text, ''), coalesce(model_id::text, ''), kind,
		CASE WHEN boost_until > now() AND boost_value IS NOT NULL THEN boost_value ELSE value END FROM limits
		WHERE (model_id IS NULL OR model_id = $5::uuid) AND (scope_type = 'org'
			OR (scope_type IN ('project', 'member') AND scope_id = $1::uuid)
			OR (scope_type = 'user' AND scope_id = $2::uuid)
			OR (scope_type = 'service_account' AND scope_id = $3::uuid)
			OR (scope_type = 'api_key' AND scope_id = $4::uuid))`,
		nullID(s.ProjectID), nullID(s.UserID), nullID(s.ServiceAccountID), nullID(s.KeyID), nullID(s.ModelID))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Limit
	for rows.Next() {
		var l Limit
		if err := rows.Scan(&l.ScopeType, &l.ScopeID, &l.ModelID, &l.Kind, &l.Value); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, rows.Err()
}

func nullID(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func scopeKey(l Limit, s Subject) string {
	var k string
	switch l.ScopeType {
	case "org":
		k = "org"
	case "project":
		k = "project:" + l.ScopeID
	case "member":
		if s.UserID == "" {
			return ""
		}
		k = "member:" + l.ScopeID + ":" + s.UserID
	case "user":
		k = "user:" + l.ScopeID
	case "service_account":
		k = "sa:" + l.ScopeID
	case "api_key":
		k = "key:" + l.ScopeID
	default:
		return ""
	}
	if l.ModelID != "" {
		k += "/model:" + l.ModelID
	}
	return k
}

// Build turns limits into the checks admission runs, reserving tokens and micros against budgets.
func Build(limits []Limit, s Subject, now time.Time, tokens, micros int64) (scopes []Scope, rates, budgets []Counter) {
	minute := now.UTC().Truncate(time.Minute)
	day := time.Date(now.UTC().Year(), now.UTC().Month(), now.UTC().Day(), 0, 0, 0, 0, time.UTC)
	month := time.Date(now.UTC().Year(), now.UTC().Month(), 1, 0, 0, 0, 0, time.UTC)
	for _, l := range limits {
		k := scopeKey(l, s)
		if k == "" {
			continue
		}
		switch l.Kind {
		case "concurrency":
			scopes = append(scopes, Scope{Key: "conc:" + k, Limit: l.Value})
		case "requests_per_minute":
			rates = append(rates, Counter{Key: "rpm:" + k, Window: minute, Limit: l.Value, Amount: 1})
		case "tokens_per_minute":
			rates = append(rates, Counter{Key: "tpm:" + k, Window: minute, Limit: l.Value, Amount: tokens})
		case "tokens_per_day":
			budgets = append(budgets, Counter{Key: "tokens:day:" + k, Window: day, Limit: l.Value, Amount: tokens})
		case "tokens_per_month":
			budgets = append(budgets, Counter{Key: "tokens:month:" + k, Window: month, Limit: l.Value, Amount: tokens})
		case "cost_per_day":
			budgets = append(budgets, Counter{Key: "cost:day:" + k, Window: day, Limit: l.Value, Amount: micros})
		case "cost_per_month":
			budgets = append(budgets, Counter{Key: "cost:month:" + k, Window: month, Limit: l.Value, Amount: micros})
		}
	}
	if s.TaskID != "" {
		scopes = append(scopes, Scope{Key: "conc:task:" + s.TaskID, Limit: int64(max(1, s.TaskConcurrency))})
		if s.TreeConcurrency > 0 {
			scopes = append(scopes, Scope{Key: "conc:tree:" + s.TaskRootID, Limit: int64(s.TreeConcurrency)})
		}
		if s.TaskBudget != nil {
			budgets = append(budgets, Counter{Key: "tokens:task:" + s.TaskRootID, Window: time.Unix(0, 0).UTC(), Limit: *s.TaskBudget, Amount: tokens})
		}
	}
	return scopes, rates, budgets
}

func isTPM(key string) bool { return strings.HasPrefix(key, "tpm:") }

func isCost(key string) bool { return strings.HasPrefix(key, "cost:") }

func windowLength(key string) time.Duration {
	switch {
	case strings.HasPrefix(key, "rpm:"), strings.HasPrefix(key, "tpm:"):
		return time.Minute
	case strings.Contains(key, ":day:"):
		return 24 * time.Hour
	case strings.Contains(key, ":month:"):
		return 31 * 24 * time.Hour
	default:
		return 24 * time.Hour
	}
}
