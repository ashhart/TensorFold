package fleet

import (
	"context"
	"encoding/json"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

// Capabilities describe what a model can do, shown to clients and used to pick endpoints.
type Capabilities struct {
	ContextWindow int  `json:"context_window,omitempty"`
	Tools         bool `json:"tools,omitempty"`
	Vision        bool `json:"vision,omitempty"`
	Reasoning     bool `json:"reasoning,omitempty"`
}

// Prices are micro-dollars per million tokens.
type Prices struct {
	Input  int64 `json:"input_micros"`
	Output int64 `json:"output_micros"`
	Cached int64 `json:"cached_micros"`
}

// Model is one catalogue entry.
type Model struct {
	ID           string
	Name         string
	DisplayName  string
	Description  string
	Kind         string
	RoutePolicy  string
	Status       string
	Capabilities Capabilities
	Chargeback   Prices
	Deployments  int
	Routes       int
	Grants       int
	CreatedAt    time.Time
}

// ModelInput is what the model form or configuration file sets.
type ModelInput struct {
	Name         string
	DisplayName  string
	Description  string
	Kind         string
	RoutePolicy  string
	Status       string
	Capabilities Capabilities
	Chargeback   Prices
}

func (in *ModelInput) validate() error {
	if in.Kind == "" {
		in.Kind = "chat"
	}
	if in.RoutePolicy == "" {
		in.RoutePolicy = "local_first"
	}
	if in.Status == "" {
		in.Status = "active"
	}
	if in.DisplayName == "" {
		in.DisplayName = in.Name
	}
	if !modelPattern.MatchString(in.Name) {
		return Invalid("model names are up to 128 letters, digits and . _ : / -")
	}
	if in.Capabilities.ContextWindow < 0 || in.Chargeback.Input < 0 || in.Chargeback.Output < 0 || in.Chargeback.Cached < 0 {
		return Invalid("context window and prices cannot be negative")
	}
	for _, err := range []error{text("display name", in.DisplayName, 120, true), text("description", in.Description, 2000, false),
		oneOf("kind", in.Kind, "chat", "embedding"), oneOf("route policy", in.RoutePolicy, "local_only", "local_first", "cloud_first", "cloud_only"),
		oneOf("status", in.Status, "active", "disabled")} {
		if err != nil {
			return err
		}
	}
	return nil
}

const modelCols = `m.id, m.name, m.display_name, m.description, m.kind, m.route_policy, m.status, m.capabilities, m.chargeback, m.created_at,
	(SELECT count(*) FROM deployments d WHERE d.model_id = m.id AND d.status <> 'retired'),
	(SELECT count(*) FROM provider_routes r WHERE r.model_id = m.id), (SELECT count(*) FROM model_grants g WHERE g.model_id = m.id)`

func scanModel(row interface{ Scan(...any) error }) (Model, error) {
	var m Model
	var caps, charge []byte
	err := row.Scan(&m.ID, &m.Name, &m.DisplayName, &m.Description, &m.Kind, &m.RoutePolicy, &m.Status, &caps, &charge, &m.CreatedAt,
		&m.Deployments, &m.Routes, &m.Grants)
	if err != nil {
		return m, notFound(err)
	}
	_ = json.Unmarshal(caps, &m.Capabilities)
	_ = json.Unmarshal(charge, &m.Chargeback)
	return m, nil
}

// Models lists the catalogue.
func Models(ctx context.Context, q db.Querier) ([]Model, error) {
	rows, err := q.Query(ctx, "SELECT "+modelCols+" FROM models m ORDER BY m.name")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Model
	for rows.Next() {
		m, err := scanModel(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// ModelByName loads one model.
func ModelByName(ctx context.Context, q db.Querier, name string) (Model, error) {
	return scanModel(q.QueryRow(ctx, "SELECT "+modelCols+" FROM models m WHERE m.name = $1", name))
}

// ModelByID loads one model by id; console URLs use ids because model names may contain slashes.
func ModelByID(ctx context.Context, q db.Querier, id string) (Model, error) {
	if !ids.Valid(id) {
		return Model{}, ErrNotFound
	}
	return scanModel(q.QueryRow(ctx, "SELECT "+modelCols+" FROM models m WHERE m.id = $1", id))
}

// CreateModel adds a catalogue model.
func (t *Tx) CreateModel(ctx context.Context, in ModelInput) (Model, error) {
	if err := in.validate(); err != nil {
		return Model{}, err
	}
	caps, _ := json.Marshal(in.Capabilities)
	charge, _ := json.Marshal(in.Chargeback)
	id := ids.New()
	_, err := t.tx.Exec(ctx, `INSERT INTO models (id, name, display_name, description, kind, route_policy, status, capabilities, chargeback)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`, id, in.Name, in.DisplayName, in.Description, in.Kind, in.RoutePolicy, in.Status, caps, charge)
	if err != nil {
		return Model{}, uniqueMsg(err, "a model with that name already exists")
	}
	if err := t.record(ctx, "model.created", "model", id, map[string]any{"name": in.Name, "route_policy": in.RoutePolicy}); err != nil {
		return Model{}, err
	}
	return ModelByName(ctx, t.tx, in.Name)
}

// UpdateModel changes everything but the name.
func (t *Tx) UpdateModel(ctx context.Context, id string, in ModelInput) (Model, error) {
	if err := in.validate(); err != nil {
		return Model{}, err
	}
	caps, _ := json.Marshal(in.Capabilities)
	charge, _ := json.Marshal(in.Chargeback)
	tag, err := t.tx.Exec(ctx, `UPDATE models SET display_name = $2, description = $3, kind = $4, route_policy = $5, status = $6, capabilities = $7,
		chargeback = $8, updated_at = now() WHERE id = $1`, id, in.DisplayName, in.Description, in.Kind, in.RoutePolicy, in.Status, caps, charge)
	if err != nil {
		return Model{}, err
	}
	if tag.RowsAffected() == 0 {
		return Model{}, ErrNotFound
	}
	if err := t.record(ctx, "model.updated", "model", id, map[string]any{"route_policy": in.RoutePolicy, "status": in.Status,
		"chargeback": in.Chargeback}); err != nil {
		return Model{}, err
	}
	return scanModel(t.tx.QueryRow(ctx, "SELECT "+modelCols+" FROM models m WHERE m.id = $1", id))
}

// Route is a frontier provider serving a model.
type Route struct {
	ID             string
	ModelID        string
	ProviderID     string
	ProviderName   string
	ProviderKind   string
	UpstreamModel  string
	Protocols      []string
	Priority       int
	Prices         Prices
	MaxConcurrency int
	Enabled        bool
}

// RouteInput is what the route form or configuration file sets.
type RouteInput struct {
	ProviderID     string
	UpstreamModel  string
	Protocols      []string
	Priority       int
	Prices         Prices
	MaxConcurrency int
	Enabled        bool
}

var knownProtocols = map[string]bool{providers.ChatCompletions: true, providers.Completions: true, providers.Embeddings: true,
	providers.Responses: true, providers.Messages: true, providers.CountTokens: true}

func (in *RouteInput) validate(kind string) error {
	if !modelPattern.MatchString(in.UpstreamModel) {
		return Invalid("upstream model must be the provider's model name")
	}
	if len(in.Protocols) == 0 {
		in.Protocols = providers.DefaultProtocols[kind]
	}
	for _, p := range in.Protocols {
		if !knownProtocols[p] {
			return Invalid("unknown protocol " + p)
		}
	}
	if in.Priority == 0 {
		in.Priority = 100
	}
	if in.MaxConcurrency <= 0 {
		in.MaxConcurrency = 64
	}
	if in.Priority < 0 || in.Priority > 10000 || in.MaxConcurrency > 10000 {
		return Invalid("priority and concurrency must be between 1 and 10000")
	}
	if in.Prices.Input < 0 || in.Prices.Output < 0 || in.Prices.Cached < 0 {
		return Invalid("prices cannot be negative")
	}
	return nil
}

// Routes lists provider routes for a model, or all when modelID is empty.
func Routes(ctx context.Context, q db.Querier, modelID string) ([]Route, error) {
	rows, err := q.Query(ctx, `SELECT r.id, r.model_id, r.provider_id, p.name, p.kind, r.upstream_model, r.protocols, r.priority, r.price_input_micros,
		r.price_output_micros, r.price_cached_micros, r.max_concurrency, r.enabled FROM provider_routes r JOIN providers p ON p.id = r.provider_id
		WHERE $1 = '' OR r.model_id::text = $1 ORDER BY r.priority, p.name`, modelID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Route
	for rows.Next() {
		var r Route
		if err := rows.Scan(&r.ID, &r.ModelID, &r.ProviderID, &r.ProviderName, &r.ProviderKind, &r.UpstreamModel, &r.Protocols, &r.Priority,
			&r.Prices.Input, &r.Prices.Output, &r.Prices.Cached, &r.MaxConcurrency, &r.Enabled); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// SetRoute creates or updates the route from a model to a provider's upstream model.
func (t *Tx) SetRoute(ctx context.Context, modelID string, in RouteInput) error {
	var kind string
	if err := t.tx.QueryRow(ctx, "SELECT kind FROM providers WHERE id = $1", in.ProviderID).Scan(&kind); err != nil {
		return notFound(err)
	}
	if err := in.validate(kind); err != nil {
		return err
	}
	var id string
	err := t.tx.QueryRow(ctx, `INSERT INTO provider_routes (id, model_id, provider_id, upstream_model, protocols, priority, price_input_micros,
		price_output_micros, price_cached_micros, max_concurrency, enabled) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
		ON CONFLICT (model_id, provider_id, upstream_model) DO UPDATE SET protocols = EXCLUDED.protocols, priority = EXCLUDED.priority,
		price_input_micros = EXCLUDED.price_input_micros, price_output_micros = EXCLUDED.price_output_micros,
		price_cached_micros = EXCLUDED.price_cached_micros, max_concurrency = EXCLUDED.max_concurrency, enabled = EXCLUDED.enabled
		RETURNING id`, ids.New(), modelID, in.ProviderID, in.UpstreamModel, in.Protocols, in.Priority, in.Prices.Input, in.Prices.Output,
		in.Prices.Cached, in.MaxConcurrency, in.Enabled).Scan(&id)
	if db.IsForeignKey(err) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	return t.record(ctx, "model.route_set", "model", modelID, map[string]any{"route_id": id, "provider_id": in.ProviderID,
		"upstream_model": in.UpstreamModel, "protocols": in.Protocols, "enabled": in.Enabled, "prices": in.Prices})
}

// RemoveRoute deletes a route.
func (t *Tx) RemoveRoute(ctx context.Context, id string) error {
	var modelID string
	if err := t.tx.QueryRow(ctx, "DELETE FROM provider_routes WHERE id = $1 RETURNING model_id", id).Scan(&modelID); err != nil {
		return notFound(err)
	}
	return t.record(ctx, "model.route_removed", "model", modelID, map[string]any{"route_id": id})
}
