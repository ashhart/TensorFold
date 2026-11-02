package fleet

import (
	"context"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Project is a unit of access and budget.
type Project struct {
	ID             string
	Slug           string
	Name           string
	Description    string
	Classification string
	MaxPriority    string
	Everyone       bool
	PlanApproval   string
	SecretsGuard   string
	IsolateCache   bool
	ArchivedAt     *time.Time
	CreatedAt      time.Time
}

// ProjectInput is what a project's settings form or configuration file sets.
type ProjectInput struct {
	Slug           string
	Name           string
	Description    string
	Classification string
	MaxPriority    string
	Everyone       bool
	PlanApproval   string
	SecretsGuard   string
	IsolateCache   bool
}

func (in *ProjectInput) defaults() {
	if in.Classification == "" {
		in.Classification = "internal"
	}
	if in.MaxPriority == "" {
		in.MaxPriority = "interactive"
	}
	if in.PlanApproval == "" {
		in.PlanApproval = "required"
	}
	if in.SecretsGuard == "" {
		in.SecretsGuard = "local"
	}
}

func (in ProjectInput) validate() error {
	for _, err := range []error{slug("slug", in.Slug), text("name", in.Name, 120, true), text("description", in.Description, 2000, false),
		class(in.Classification), priority(in.MaxPriority), oneOf("plan approval", in.PlanApproval, "required", "auto"),
		oneOf("credential guard", in.SecretsGuard, "local", "block", "off")} {
		if err != nil {
			return err
		}
	}
	return nil
}

const projectCols = `id, slug, name, description, classification, max_priority, everyone, plan_approval, secrets_guard, isolate_prompt_cache,
	archived_at, created_at`

func scanProject(row interface{ Scan(...any) error }) (Project, error) {
	var p Project
	err := row.Scan(&p.ID, &p.Slug, &p.Name, &p.Description, &p.Classification, &p.MaxPriority, &p.Everyone, &p.PlanApproval, &p.SecretsGuard,
		&p.IsolateCache, &p.ArchivedAt, &p.CreatedAt)
	return p, notFound(err)
}

// Projects lists projects; a nil filter lists all, otherwise only those ids.
func Projects(ctx context.Context, q db.Querier, only []string) ([]Project, error) {
	sql := "SELECT " + projectCols + " FROM projects"
	args := []any{}
	if only != nil {
		sql += " WHERE id = ANY($1::uuid[])"
		args = append(args, only)
	}
	rows, err := q.Query(ctx, sql+" ORDER BY archived_at NULLS FIRST, slug", args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Project
	for rows.Next() {
		p, err := scanProject(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ProjectBySlug loads one project.
func ProjectBySlug(ctx context.Context, q db.Querier, s string) (Project, error) {
	return scanProject(q.QueryRow(ctx, "SELECT "+projectCols+" FROM projects WHERE slug = $1", s))
}

// CreateProject adds a project.
func (t *Tx) CreateProject(ctx context.Context, in ProjectInput) (Project, error) {
	in.defaults()
	if err := in.validate(); err != nil {
		return Project{}, err
	}
	p, err := scanProject(t.tx.QueryRow(ctx, `INSERT INTO projects (id, slug, name, description, classification, max_priority, everyone,
		plan_approval, secrets_guard, isolate_prompt_cache) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING `+projectCols,
		ids.New(), in.Slug, in.Name, in.Description, in.Classification, in.MaxPriority, in.Everyone, in.PlanApproval, in.SecretsGuard, in.IsolateCache))
	if err != nil {
		return p, uniqueMsg(err, "a project with that slug already exists")
	}
	return p, t.record(ctx, "project.created", "project", p.ID, map[string]any{"slug": p.Slug, "classification": p.Classification})
}

// UpdateProject changes a project's settings; its slug is fixed.
func (t *Tx) UpdateProject(ctx context.Context, id string, in ProjectInput) (Project, error) {
	in.defaults()
	if err := in.validate(); err != nil {
		return Project{}, err
	}
	p, err := scanProject(t.tx.QueryRow(ctx, `UPDATE projects SET name = $2, description = $3, classification = $4, max_priority = $5,
		everyone = $6, plan_approval = $7, secrets_guard = $8, isolate_prompt_cache = $9, updated_at = now() WHERE id = $1 RETURNING `+projectCols,
		id, in.Name, in.Description, in.Classification, in.MaxPriority, in.Everyone, in.PlanApproval, in.SecretsGuard, in.IsolateCache))
	if err != nil {
		return p, err
	}
	return p, t.record(ctx, "project.updated", "project", p.ID, map[string]any{"slug": p.Slug, "classification": p.Classification,
		"max_priority": p.MaxPriority, "everyone": p.Everyone, "plan_approval": p.PlanApproval, "secrets_guard": p.SecretsGuard,
		"isolate_prompt_cache": p.IsolateCache})
}

// SetArchived archives or restores a project; archived projects grant nothing.
func (t *Tx) SetArchived(ctx context.Context, id string, archived bool) error {
	tag, err := t.tx.Exec(ctx, "UPDATE projects SET archived_at = CASE WHEN $2 THEN now() ELSE NULL END, updated_at = now() WHERE id = $1", id, archived)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	action := "project.restored"
	if archived {
		action = "project.archived"
	}
	return t.record(ctx, action, "project", id, nil)
}

// Grant lets a project, or every project when ProjectID is empty, use a model within caps.
type Grant struct {
	ID              string
	ModelID         string
	ModelName       string
	ProjectID       string
	ProjectSlug     string
	MaxContext      *int
	MaxOutputTokens *int
}

// Grants lists grants for a project, or every grant when projectID is empty.
func Grants(ctx context.Context, q db.Querier, projectID, modelID string) ([]Grant, error) {
	rows, err := q.Query(ctx, `SELECT g.id, g.model_id, m.name, coalesce(g.project_id::text, ''), coalesce(p.slug, ''), g.max_context, g.max_output_tokens
		FROM model_grants g JOIN models m ON m.id = g.model_id LEFT JOIN projects p ON p.id = g.project_id
		WHERE ($1 = '' OR g.project_id::text = $1) AND ($2 = '' OR g.model_id::text = $2) ORDER BY m.name, p.slug NULLS FIRST`, projectID, modelID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Grant
	for rows.Next() {
		var g Grant
		if err := rows.Scan(&g.ID, &g.ModelID, &g.ModelName, &g.ProjectID, &g.ProjectSlug, &g.MaxContext, &g.MaxOutputTokens); err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, rows.Err()
}

// SetGrant creates or replaces the grant of a model to a project, or to every project when projectID is empty.
func (t *Tx) SetGrant(ctx context.Context, modelID, projectID string, maxContext, maxOutput *int) error {
	if maxContext != nil && *maxContext <= 0 || maxOutput != nil && *maxOutput <= 0 {
		return Invalid("caps must be positive token counts, or empty for no cap")
	}
	// A model only an isolated project used has replicas holding its prefixes, which a new grantee must not reach.
	var private bool
	if err := t.tx.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM model_grants g JOIN projects p ON p.id = g.project_id WHERE g.model_id = $1
		AND p.isolate_prompt_cache AND g.project_id IS DISTINCT FROM $2::uuid
		AND NOT EXISTS (SELECT 1 FROM model_grants o WHERE o.model_id = $1 AND o.project_id IS DISTINCT FROM g.project_id))`,
		modelID, optID(projectID)).Scan(&private); err != nil {
		return err
	}
	var id string
	err := t.tx.QueryRow(ctx, `INSERT INTO model_grants (id, model_id, project_id, max_context, max_output_tokens) VALUES ($1,$2,$3,$4,$5)
		ON CONFLICT (model_id, project_id) DO UPDATE SET max_context = EXCLUDED.max_context, max_output_tokens = EXCLUDED.max_output_tokens
		RETURNING id`, ids.New(), modelID, optID(projectID), maxContext, maxOutput).Scan(&id)
	if db.IsForeignKey(err) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	if private {
		if err := t.stopReplicas(ctx, sharedCaches, modelID); err != nil {
			return err
		}
	}
	return t.record(ctx, "model.granted", "model", modelID, map[string]any{"project_id": projectID, "max_context": maxContext, "max_output_tokens": maxOutput})
}

// sharedCaches selects a model's replicas that keep prompt states, which may hold a project's prefixes.
const sharedCaches = "deployment_id IN (SELECT id FROM deployments WHERE model_id = $1 AND spec->>'prompt_cache' IS DISTINCT FROM 'off')"

// RemoveGrant withdraws one grant; when an isolated project loses a model only it used, its replicas restart empty.
func (t *Tx) RemoveGrant(ctx context.Context, id string) error {
	var modelID string
	var project *string
	var private bool
	err := t.tx.QueryRow(ctx, `WITH g AS (DELETE FROM model_grants WHERE id = $1 RETURNING model_id, project_id)
		SELECT g.model_id, g.project_id, coalesce(p.isolate_prompt_cache, false)
			AND NOT EXISTS (SELECT 1 FROM model_grants o WHERE o.model_id = g.model_id AND o.id <> $1)
		FROM g LEFT JOIN projects p ON p.id = g.project_id`, id).Scan(&modelID, &project, &private)
	if err != nil {
		return notFound(err)
	}
	if private {
		if err := t.stopReplicas(ctx, sharedCaches, modelID); err != nil {
			return err
		}
	}
	return t.record(ctx, "model.grant_removed", "model", modelID, map[string]any{"project_id": project})
}

// Alias is a requested name that resolves to a catalogue model, for one project or globally.
type Alias struct {
	Alias       string
	ProjectID   string
	ProjectSlug string
	ModelID     string
	ModelName   string
}

// Aliases lists aliases for a project, global aliases when projectID is "global", or all when empty.
func Aliases(ctx context.Context, q db.Querier, projectID string) ([]Alias, error) {
	rows, err := q.Query(ctx, `SELECT a.alias, coalesce(a.project_id::text, ''), coalesce(p.slug, ''), a.model_id, m.name
		FROM model_aliases a JOIN models m ON m.id = a.model_id LEFT JOIN projects p ON p.id = a.project_id
		WHERE $1 = '' OR ($1 = 'global' AND a.project_id IS NULL) OR a.project_id::text = $1 ORDER BY a.alias, p.slug NULLS FIRST`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Alias
	for rows.Next() {
		var a Alias
		if err := rows.Scan(&a.Alias, &a.ProjectID, &a.ProjectSlug, &a.ModelID, &a.ModelName); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

// SetAlias points alias at a model, in one project or globally when projectID is empty.
func (t *Tx) SetAlias(ctx context.Context, alias, projectID, modelID string) error {
	if !aliasPattern.MatchString(alias) {
		return Invalid("aliases are up to 128 letters, digits and . _ : / @ -")
	}
	_, err := t.tx.Exec(ctx, `INSERT INTO model_aliases (alias, project_id, model_id) VALUES ($1, $2, $3)
		ON CONFLICT (alias, project_id) DO UPDATE SET model_id = EXCLUDED.model_id`, alias, optID(projectID), modelID)
	if db.IsForeignKey(err) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	return t.record(ctx, "model.alias_set", "model", modelID, map[string]any{"alias": alias, "project_id": projectID})
}

// RemoveAlias deletes an alias.
func (t *Tx) RemoveAlias(ctx context.Context, alias, projectID string) error {
	var modelID string
	err := t.tx.QueryRow(ctx, "DELETE FROM model_aliases WHERE alias = $1 AND project_id IS NOT DISTINCT FROM $2 RETURNING model_id",
		alias, optID(projectID)).Scan(&modelID)
	if err != nil {
		return notFound(err)
	}
	return t.record(ctx, "model.alias_removed", "model", modelID, map[string]any{"alias": alias, "project_id": projectID})
}
