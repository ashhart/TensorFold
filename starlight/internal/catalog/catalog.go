// Package catalog resolves requested model names through aliases and grants to catalogue models and their routes.
package catalog

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Model is one catalogue entry as the gateway needs it.
type Model struct {
	ID           string
	Name         string
	DisplayName  string
	Kind         string
	RoutePolicy  string
	Capabilities map[string]any
	Chargeback   Prices
}

// Prices are micro-dollars per million tokens.
type Prices struct {
	Input  int64 `json:"input_micros"`
	Output int64 `json:"output_micros"`
	Cached int64 `json:"cached_micros"`
}

// Cost prices a request's usage, saturating rather than wrapping when absurd prices meet large counts.
func (p Prices) Cost(prompt, completion, cached int64) int64 {
	total := mulAdd(0, max(prompt-cached, 0), p.Input)
	total = mulAdd(total, cached, p.Cached)
	return mulAdd(total, completion, p.Output) / 1_000_000
}

// mulAdd returns acc + a*b for a non-negative acc, treating negative factors as zero and saturating at the int64 limit.
func mulAdd(acc, a, b int64) int64 {
	if a <= 0 || b <= 0 {
		return acc
	}
	if a > (math.MaxInt64-acc)/b {
		return math.MaxInt64
	}
	return acc + a*b
}

// Grant caps a project's use of a model.
type Grant struct {
	MaxContext      int
	MaxOutputTokens int
}

// ProviderRoute is a frontier provider serving a model.
type ProviderRoute struct {
	ID             string
	ProviderID     string
	ProviderName   string
	Kind           string
	BaseURL        string
	UpstreamModel  string
	Protocols      []string
	Priority       int
	MaxConcurrency int
	MaxClass       string
	Region         string
	Prices         Prices
}

// Speaks reports whether the route accepts a protocol such as "openai.chat" or "anthropic.messages".
func (r ProviderRoute) Speaks(protocol string) bool {
	for _, p := range r.Protocols {
		if p == protocol {
			return true
		}
	}
	return false
}

// ErrNotFound covers both unknown and ungranted models, so callers cannot probe the catalogue.
var ErrNotFound = errors.New("catalog: model not found")

// Catalog caches resolutions for a few seconds; catalogue edits take effect within TTL.
type Catalog struct {
	Pool *pgxpool.Pool
	TTL  time.Duration
	mu   sync.Mutex
	hits map[string]entry
}

type entry struct {
	model   Model
	grant   Grant
	err     error
	expires time.Time
}

// Resolve maps name, acting in project, to a model the project is granted.
func (c *Catalog) Resolve(ctx context.Context, projectID, name string) (Model, Grant, error) {
	key := projectID + "\x00" + name
	c.mu.Lock()
	if e, ok := c.hits[key]; ok && time.Now().Before(e.expires) {
		c.mu.Unlock()
		return e.model, e.grant, e.err
	}
	c.mu.Unlock()
	m, g, err := c.resolve(ctx, projectID, name)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return m, g, err
	}
	ttl := c.TTL
	if ttl <= 0 {
		ttl = 5 * time.Second
	}
	c.mu.Lock()
	if c.hits == nil || len(c.hits) > 50000 {
		c.hits = map[string]entry{}
	}
	c.hits[key] = entry{m, g, err, time.Now().Add(ttl)}
	c.mu.Unlock()
	return m, g, err
}

func (c *Catalog) resolve(ctx context.Context, projectID, name string) (Model, Grant, error) {
	var m Model
	var caps, charge []byte
	var maxCtx, maxOut *int
	err := c.Pool.QueryRow(ctx, `WITH target AS (
			SELECT model_id, 1 AS pri FROM model_aliases WHERE alias = $2 AND project_id = $1
			UNION ALL SELECT model_id, 2 FROM model_aliases WHERE alias = $2 AND project_id IS NULL
			UNION ALL SELECT id, 3 FROM models WHERE name = $2 ORDER BY pri LIMIT 1)
		SELECT m.id, m.name, m.display_name, m.kind, m.route_policy, m.capabilities, m.chargeback, g.max_context, g.max_output_tokens
		FROM target t JOIN models m ON m.id = t.model_id
		JOIN LATERAL (SELECT max_context, max_output_tokens FROM model_grants
			WHERE model_id = m.id AND (project_id = $1 OR project_id IS NULL) ORDER BY project_id NULLS LAST LIMIT 1) g ON true
		WHERE m.status = 'active'`, projectID, name).
		Scan(&m.ID, &m.Name, &m.DisplayName, &m.Kind, &m.RoutePolicy, &caps, &charge, &maxCtx, &maxOut)
	if db.NotFound(err) {
		return Model{}, Grant{}, ErrNotFound
	}
	if err != nil {
		return Model{}, Grant{}, err
	}
	_ = json.Unmarshal(caps, &m.Capabilities)
	_ = json.Unmarshal(charge, &m.Chargeback)
	var g Grant
	if maxCtx != nil {
		g.MaxContext = *maxCtx
	}
	if maxOut != nil {
		g.MaxOutputTokens = *maxOut
	}
	return m, g, nil
}

// Visible lists model names a project may use, for /v1/models.
func (c *Catalog) Visible(ctx context.Context, projectID string) ([]Model, error) {
	rows, err := c.Pool.Query(ctx, `SELECT DISTINCT m.id, m.name, m.display_name, m.kind, m.route_policy FROM models m
		JOIN model_grants g ON g.model_id = m.id AND (g.project_id = $1 OR g.project_id IS NULL)
		WHERE m.status = 'active' ORDER BY m.name`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Model
	for rows.Next() {
		var m Model
		if err := rows.Scan(&m.ID, &m.Name, &m.DisplayName, &m.Kind, &m.RoutePolicy); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

const routeCols = `SELECT r.id, p.id, p.name, p.kind, p.base_url, r.upstream_model, r.protocols, r.priority, r.max_concurrency,
	p.max_classification, p.region, r.price_input_micros, r.price_output_micros, r.price_cached_micros
	FROM provider_routes r JOIN providers p ON p.id = r.provider_id`

// Route loads one provider route by id, enabled or not, for reading responses it stored.
func (c *Catalog) Route(ctx context.Context, id string) (ProviderRoute, error) {
	rows, err := c.Pool.Query(ctx, routeCols+" WHERE r.id = $1", id)
	if err != nil {
		return ProviderRoute{}, err
	}
	defer rows.Close()
	routes, err := scanRoutes(rows)
	if err != nil || len(routes) == 0 {
		return ProviderRoute{}, ErrNotFound
	}
	return routes[0], nil
}

// ProviderRoutes lists enabled provider routes for a model in priority order.
func (c *Catalog) ProviderRoutes(ctx context.Context, modelID string) ([]ProviderRoute, error) {
	rows, err := c.Pool.Query(ctx, routeCols+" WHERE r.model_id = $1 AND r.enabled AND p.enabled ORDER BY r.priority, r.id", modelID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanRoutes(rows)
}

func scanRoutes(rows pgx.Rows) ([]ProviderRoute, error) {
	var out []ProviderRoute
	for rows.Next() {
		var r ProviderRoute
		if err := rows.Scan(&r.ID, &r.ProviderID, &r.ProviderName, &r.Kind, &r.BaseURL, &r.UpstreamModel, &r.Protocols, &r.Priority,
			&r.MaxConcurrency, &r.MaxClass, &r.Region, &r.Prices.Input, &r.Prices.Output, &r.Prices.Cached); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// Flush forgets cached resolutions after an edit made by this instance.
func (c *Catalog) Flush() {
	c.mu.Lock()
	c.hits = nil
	c.mu.Unlock()
}
