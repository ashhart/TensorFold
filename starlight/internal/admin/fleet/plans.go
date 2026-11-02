package fleet

import (
	"context"
	"encoding/hex"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Plan is a frontier-drafted plan as the console shows it; Sealed stays encrypted until a permitted viewer opens it.
type Plan struct {
	ID             string
	ProjectID      string
	ProjectSlug    string
	OwnerType      string
	OwnerID        string
	Owner          string
	Title          string
	Planner        string
	Executor       string
	Status         string
	Error          string
	EgressSHA256   string
	EgressBytes    int
	PlanSHA256     string
	Sealed         []byte
	BudgetTokens   *int64
	MaxConcurrency int
	MaxDepth       int
	TTLSeconds     int
	ApprovedBy     string
	ApprovedAt     *time.Time
	CreatedAt      time.Time
}

// PlanFilter narrows plans; nil Projects, empty Statuses and empty OwnerID leave that dimension unfiltered.
type PlanFilter struct {
	Projects []string
	Statuses []string
	OwnerID  string
	Limit    int
}

const planCols = `pl.id, pl.project_id, p.slug, pl.owner_type, pl.owner_id, coalesce(u.email, sa.name, ''), pl.title, pl.planner_model,
	pl.executor_model, pl.status, pl.error, pl.egress_sha256, pl.egress_bytes, coalesce(pl.plan_sha256, ''::bytea), pl.plan_sealed,
	pl.budget_tokens, pl.max_concurrency, pl.max_depth, pl.ttl_seconds, coalesce(ap.email, ''), pl.approved_at, pl.created_at
	FROM plans pl JOIN projects p ON p.id = pl.project_id
	LEFT JOIN users u ON pl.owner_type = 'user' AND u.id = pl.owner_id
	LEFT JOIN service_accounts sa ON pl.owner_type = 'service_account' AND sa.id = pl.owner_id
	LEFT JOIN users ap ON ap.id = pl.approved_by`

func scanPlan(row interface{ Scan(...any) error }) (Plan, error) {
	var p Plan
	var egress, sum []byte
	err := row.Scan(&p.ID, &p.ProjectID, &p.ProjectSlug, &p.OwnerType, &p.OwnerID, &p.Owner, &p.Title, &p.Planner, &p.Executor, &p.Status,
		&p.Error, &egress, &p.EgressBytes, &sum, &p.Sealed, &p.BudgetTokens, &p.MaxConcurrency, &p.MaxDepth, &p.TTLSeconds, &p.ApprovedBy,
		&p.ApprovedAt, &p.CreatedAt)
	if err != nil {
		return p, notFound(err)
	}
	p.EgressSHA256 = hex.EncodeToString(egress)
	if len(sum) > 0 {
		p.PlanSHA256 = hex.EncodeToString(sum)
	}
	return p, nil
}

// Plans lists plans newest first.
func Plans(ctx context.Context, q db.Querier, f PlanFilter) ([]Plan, error) {
	if f.Limit <= 0 {
		f.Limit = 100
	}
	projects := f.Projects
	if projects == nil {
		projects = []string{}
	}
	rows, err := q.Query(ctx, "SELECT "+planCols+` WHERE ($1 OR pl.project_id = ANY($2::uuid[])) AND (coalesce(cardinality($3::text[]), 0) = 0 OR pl.status = ANY($3))
		AND ($4 = '' OR pl.owner_type = 'user' AND pl.owner_id::text = $4) ORDER BY pl.created_at DESC LIMIT $5`,
		f.Projects == nil, projects, f.Statuses, f.OwnerID, f.Limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Plan
	for rows.Next() {
		p, err := scanPlan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// PlanByID loads one plan.
func PlanByID(ctx context.Context, q db.Querier, id string) (Plan, error) {
	if !ids.Valid(id) {
		return Plan{}, ErrNotFound
	}
	return scanPlan(q.QueryRow(ctx, "SELECT "+planCols+" WHERE pl.id = $1", id))
}
