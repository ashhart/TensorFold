// Package tasks issues scoped credentials for agent runs; subagents get children that never exceed their parent.
package tasks

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Limits bound what any task may ask for.
const (
	MaxDepth       = 5
	MaxConcurrency = 64
	MaxTTL         = 7 * 24 * time.Hour
)

// Spec is a request to create a task.
type Spec struct {
	Name           string   `json:"name"`
	Models         []string `json:"models"`
	LocalOnly      *bool    `json:"local_only"`
	MaxConcurrency int      `json:"max_concurrency"`
	MaxDepth       int      `json:"max_depth"`
	BudgetTokens   *int64   `json:"budget_tokens"`
	TTLSeconds     int      `json:"ttl_seconds"`
	PinRevisions   bool     `json:"pin_revisions"`
	PlanID         string   `json:"-"`
}

// Task is what the API returns; Token appears only at creation.
type Task struct {
	ID             string                 `json:"id"`
	RootID         string                 `json:"root_id"`
	ParentID       *string                `json:"parent_id"`
	Depth          int                    `json:"depth"`
	Name           string                 `json:"name"`
	Models         []string               `json:"models"`
	LocalOnly      bool                   `json:"local_only"`
	MaxConcurrency int                    `json:"max_concurrency"`
	MaxDepth       int                    `json:"max_depth"`
	BudgetTokens   *int64                 `json:"budget_tokens"`
	Pins           map[string][]authn.Pin `json:"pins"`
	ExpiresAt      time.Time              `json:"expires_at"`
	Revoked        bool                   `json:"revoked"`
	Token          string                 `json:"token,omitempty"`
	UsedTokens     int64                  `json:"used_tokens"`
}

// Service creates, reads and revokes tasks.
type Service struct {
	Pool    *pgxpool.Pool
	Auth    *authn.Authenticator
	Catalog *catalog.Catalog
	Proxies *httpx.Proxies
}

func bad(msg string) error { return httpx.Errorf(http.StatusBadRequest, "invalid_request_error", msg) }

// Create makes a root task for a user or service account, or a child when the caller is itself a task.
func (s *Service) Create(ctx context.Context, c *authn.Caller, spec Spec) (Task, error) {
	if !c.P.HasScope("inference") {
		return Task{}, httpx.ErrForbidden
	}
	if c.Task != nil {
		return s.child(ctx, c, spec)
	}
	if !c.Grants.Can(authz.Tasks, c.P.ProjectID) {
		return Task{}, httpx.ErrForbidden
	}
	if spec.MaxDepth < 0 || spec.MaxDepth > MaxDepth {
		return Task{}, bad("max_depth must be between 0 and 5")
	}
	if spec.MaxConcurrency <= 0 {
		spec.MaxConcurrency = 4
	}
	if spec.MaxConcurrency > MaxConcurrency {
		return Task{}, bad("max_concurrency must be at most 64")
	}
	if spec.BudgetTokens != nil && *spec.BudgetTokens <= 0 {
		return Task{}, bad("budget_tokens must be positive")
	}
	ttl := time.Duration(spec.TTLSeconds) * time.Second
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	if ttl > MaxTTL {
		return Task{}, bad("ttl_seconds must be at most 7 days")
	}
	if c.KeyExpires != nil && time.Now().Add(ttl).After(*c.KeyExpires) {
		ttl = time.Until(*c.KeyExpires)
	}
	local := spec.LocalOnly == nil || *spec.LocalOnly
	models, pins, err := s.models(ctx, c, spec.Models, spec.PinRevisions)
	if err != nil {
		return Task{}, err
	}
	id := ids.New()
	t := Task{ID: id, RootID: id, Name: name(spec.Name), Models: models, LocalOnly: local, MaxConcurrency: spec.MaxConcurrency,
		MaxDepth: spec.MaxDepth, BudgetTokens: spec.BudgetTokens, Pins: pins, ExpiresAt: time.Now().Add(ttl)}
	return t, s.insert(ctx, c, &t, nil, spec.PlanID)
}

func (s *Service) child(ctx context.Context, c *authn.Caller, spec Spec) (Task, error) {
	p := c.Task
	if p.Depth+1 > p.MaxDepth {
		return Task{}, httpx.Errorf(http.StatusForbidden, "delegation_depth_exceeded", "this task may not create further subtasks")
	}
	if spec.BudgetTokens != nil || spec.MaxDepth != 0 {
		return Task{}, bad("subtasks share their root's budget and depth; omit budget_tokens and max_depth")
	}
	models := spec.Models
	if len(models) == 0 {
		models = c.Models
	}
	for _, m := range models {
		if !c.AllowsModel(m) {
			return Task{}, bad("subtask models must be a subset of the parent's")
		}
	}
	maxConc := spec.MaxConcurrency
	if maxConc <= 0 || maxConc > p.MaxConcurrency {
		maxConc = p.MaxConcurrency
	}
	expires := p.ExpiresAt
	if spec.TTLSeconds > 0 && time.Now().Add(time.Duration(spec.TTLSeconds)*time.Second).Before(expires) {
		expires = time.Now().Add(time.Duration(spec.TTLSeconds) * time.Second)
	}
	local := p.LocalOnly || spec.LocalOnly == nil || *spec.LocalOnly
	pins := map[string][]authn.Pin{}
	for _, m := range models {
		if pp, ok := p.Pins[m]; ok {
			pins[m] = pp
		}
	}
	parent := p.ID
	t := Task{ID: ids.New(), RootID: p.RootID, ParentID: &parent, Depth: p.Depth + 1, Name: name(spec.Name), Models: models, LocalOnly: local,
		MaxConcurrency: maxConc, MaxDepth: p.MaxDepth, Pins: pins, ExpiresAt: expires}
	return t, s.insert(ctx, c, &t, p, "")
}

func name(n string) string {
	if n == "" {
		return "agent"
	}
	if len(n) > 120 {
		return n[:120]
	}
	return n
}

// models checks each model is usable and optionally pins those with local deployments to their current revisions.
func (s *Service) models(ctx context.Context, c *authn.Caller, names []string, pin bool) ([]string, map[string][]authn.Pin, error) {
	if len(names) == 0 || len(names) > 16 {
		return nil, nil, bad("models must list between 1 and 16 models")
	}
	pins := map[string][]authn.Pin{}
	var out []string
	for _, n := range names {
		m, _, err := s.Catalog.Resolve(ctx, c.P.ProjectID, n)
		if err != nil || !c.AllowsModel(m.Name) && !c.AllowsModel(n) {
			return nil, nil, bad("model " + n + " is not available to this project")
		}
		out = append(out, m.Name)
		if !pin {
			continue
		}
		rows, err := s.Pool.Query(ctx, "SELECT id, revision FROM deployments WHERE model_id = $1 AND status = 'active'", m.ID)
		if err != nil {
			return nil, nil, err
		}
		for rows.Next() {
			var p authn.Pin
			if err := rows.Scan(&p.Deployment, &p.Revision); err != nil {
				rows.Close()
				return nil, nil, err
			}
			pins[m.Name] = append(pins[m.Name], p)
		}
		rows.Close()
	}
	return out, pins, nil
}

func (s *Service) insert(ctx context.Context, c *authn.Caller, t *Task, parent *authn.Task, planID string) error {
	tok, hash := secret.Generate(secret.Task)
	pins, _ := json.Marshal(t.Pins)
	ancestors := []string{}
	if parent != nil {
		var err error
		if ancestors, err = s.ancestors(ctx, parent.ID); err != nil {
			return err
		}
		ancestors = append(ancestors, parent.ID)
	}
	ownerType, ownerID := string(c.P.Type), c.P.ID
	var plan, key, family *string
	if ids.Valid(planID) {
		plan = &planID
	}
	if parent == nil && c.P.KeyID != "" {
		key = &c.P.KeyID
	}
	if parent == nil && c.Family != "" {
		family = &c.Family
	}
	err := db.Tx(ctx, s.Pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `INSERT INTO tasks (id, root_id, parent_id, ancestors, depth, name, owner_type, owner_id, project_id, token_hash,
			models, pins, local_only, max_depth, max_concurrency, budget_tokens, plan_id, expires_at, api_key_id, refresh_family)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)`,
			t.ID, t.RootID, t.ParentID, ancestors, t.Depth, t.Name, ownerType, ownerID, c.P.ProjectID, hash, t.Models, pins, t.LocalOnly,
			t.MaxDepth, t.MaxConcurrency, t.BudgetTokens, plan, t.ExpiresAt, key, family); err != nil {
			return err
		}
		e := audit.Actor(c.P)
		e.Action, e.TargetType, e.TargetID = "task.created", "task", t.ID
		e.Details = map[string]any{"root": t.RootID, "depth": t.Depth, "models": t.Models, "local_only": t.LocalOnly, "project_id": c.P.ProjectID}
		return audit.Append(ctx, tx, e)
	})
	t.Token = tok
	return err
}

func (s *Service) ancestors(ctx context.Context, id string) ([]string, error) {
	var out []string
	err := s.Pool.QueryRow(ctx, "SELECT ancestors FROM tasks WHERE id = $1", id).Scan(&out)
	return out, err
}

// ErrNotFound hides tasks the caller may not see.
var ErrNotFound = errors.New("tasks: not found")
