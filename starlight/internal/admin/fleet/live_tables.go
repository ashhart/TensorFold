package fleet

import (
	"context"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// InFlight is one admitted request still running.
type InFlight struct {
	ID         string `json:"id"`
	Model      string `json:"model"`
	Who        string `json:"who"`
	WhoType    string `json:"who_type"`
	Project    string `json:"project"`
	Target     string `json:"target"`
	Route      string `json:"route"`
	Priority   string `json:"priority"`
	Client     string `json:"client"`
	AgeSeconds int    `json:"age_seconds"`
}

func inFlight(ctx context.Context, q db.Querier) ([]InFlight, error) {
	rows, err := q.Query(ctx, `SELECT l.id, m.name, l.principal_type, coalesce(u.email, sa.name, ''), coalesce(p.slug, ''),
		coalesce(n.name, pv.name, ''), CASE WHEN l.replica_id IS NOT NULL THEN 'local' ELSE 'cloud' END, l.priority, l.client, l.created_at
		FROM leases l JOIN models m ON m.id = l.model_id
		LEFT JOIN users u ON l.principal_type = 'user' AND u.id = l.principal_id
		LEFT JOIN service_accounts sa ON l.principal_type = 'service_account' AND sa.id = l.principal_id
		LEFT JOIN projects p ON p.id = l.project_id
		LEFT JOIN replicas r ON r.id = l.replica_id LEFT JOIN nodes n ON n.id = r.node_id
		LEFT JOIN provider_routes pr ON pr.id = l.route_id LEFT JOIN providers pv ON pv.id = pr.provider_id
		WHERE l.heartbeat_at > now() - interval '60 seconds' ORDER BY l.created_at LIMIT 200`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []InFlight{}
	for rows.Next() {
		var f InFlight
		var at time.Time
		if err := rows.Scan(&f.ID, &f.Model, &f.WhoType, &f.Who, &f.Project, &f.Target, &f.Route, &f.Priority, &f.Client, &at); err != nil {
			return nil, err
		}
		f.AgeSeconds = int(time.Since(at).Seconds())
		out = append(out, f)
	}
	return out, rows.Err()
}

// Connected is a principal active in the last 15 minutes.
type Connected struct {
	Who        string    `json:"who"`
	WhoType    string    `json:"who_type"`
	InFlight   int       `json:"in_flight"`
	Requests   int       `json:"requests"`
	Tokens     int64     `json:"tokens"`
	LastClient string    `json:"last_client"`
	LastSeen   time.Time `json:"last_seen"`
}

func connected(ctx context.Context, q db.Querier) ([]Connected, error) {
	rows, err := q.Query(ctx, `WITH recent AS (
			SELECT principal_type, principal_id, created_at, client, prompt_tokens + completion_tokens AS tokens FROM usage_events
			WHERE created_at > now() - interval '15 minutes'),
		live AS (SELECT principal_type, principal_id, client FROM leases WHERE heartbeat_at > now() - interval '60 seconds' AND principal_id IS NOT NULL),
		seen AS (SELECT principal_type, principal_id, created_at, client FROM recent UNION ALL SELECT principal_type, principal_id, now(), client FROM live),
		who AS (SELECT principal_type, principal_id, max(created_at) AS last_seen FROM seen GROUP BY 1, 2)
		SELECT w.principal_type, coalesce(u.email, sa.name, ''), (SELECT count(*) FROM live l WHERE l.principal_id = w.principal_id),
			(SELECT count(*) FROM recent r WHERE r.principal_id = w.principal_id),
			(SELECT coalesce(sum(r.tokens), 0)::bigint FROM recent r WHERE r.principal_id = w.principal_id),
			coalesce((SELECT s.client FROM seen s WHERE s.principal_id = w.principal_id ORDER BY s.created_at DESC LIMIT 1), ''), w.last_seen
		FROM who w LEFT JOIN users u ON w.principal_type = 'user' AND u.id = w.principal_id
		LEFT JOIN service_accounts sa ON w.principal_type = 'service_account' AND sa.id = w.principal_id
		ORDER BY w.last_seen DESC LIMIT 200`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Connected{}
	for rows.Next() {
		var c Connected
		if err := rows.Scan(&c.WhoType, &c.Who, &c.InFlight, &c.Requests, &c.Tokens, &c.LastClient, &c.LastSeen); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// NodeLive is a node row of the live view.
type NodeLive struct {
	ID                    string     `json:"id"`
	Name                  string     `json:"name"`
	State                 string     `json:"state"`
	Pool                  string     `json:"pool"`
	Backend               string     `json:"backend"`
	Accelerators          string     `json:"accelerators"`
	MemoryBytes           uint64     `json:"memory_bytes"`
	Ready                 int        `json:"ready"`
	Replicas              int        `json:"replicas"`
	TokensPerSecond       float64    `json:"tokens_per_second"`
	PromptTokensPerSecond float64    `json:"prompt_tokens_per_second"`
	Running               int        `json:"running"`
	Waiting               int        `json:"waiting"`
	KVUsage               float64    `json:"kv_usage"`
	Fresh                 bool       `json:"fresh"`
	LastSeen              *time.Time `json:"last_seen"`
}

// Row is the node as a live-view row.
func (n Node) Row() NodeLive {
	return NodeLive{ID: n.ID, Name: n.Name, State: n.State(), Pool: n.Pool, Backend: n.Facts.Backend, Accelerators: n.Accelerators(),
		MemoryBytes: n.Facts.MemoryBytes, Ready: n.Ready, Replicas: n.Replicas, TokensPerSecond: n.Live.TokensPerSecond,
		PromptTokensPerSecond: n.Live.PromptTokensPerSecond, Running: n.Live.Running, Waiting: n.Live.Waiting, KVUsage: n.Live.KVUsage,
		Fresh: n.Live.Fresh, LastSeen: n.LastSeen}
}

// ModelLive is a model row of the live view.
type ModelLive struct {
	ID              string  `json:"id"`
	Name            string  `json:"name"`
	DisplayName     string  `json:"display_name"`
	RoutePolicy     string  `json:"route_policy"`
	Ready           int     `json:"ready"`
	Replicas        int     `json:"replicas"`
	Nodes           int     `json:"nodes"`
	TokensPerSecond float64 `json:"tokens_per_second"`
	Queued          int     `json:"queued"`
	CloudRoutes     int     `json:"cloud_routes"`
	Availability    string  `json:"availability"`
}

// Available reports whether requests for the model will be served now or once it loads on demand.
func (m ModelLive) Available() bool {
	return m.Availability == "available" || m.Availability == "cloud" || m.Availability == "standby"
}

// ModelsLive returns every model's availability and load, honouring offline mode.
func ModelsLive(ctx context.Context, q db.Querier) ([]ModelLive, error) {
	offline := true
	if err := q.QueryRow(ctx, "SELECT coalesce((SELECT offline FROM settings WHERE id), true)").Scan(&offline); err != nil {
		return nil, err
	}
	return modelsLive(ctx, q, offline)
}

func modelsLive(ctx context.Context, q db.Querier, offline bool) ([]ModelLive, error) {
	rows, err := q.Query(ctx, `WITH healthy AS (SELECT r.*, d.model_id, d.mode FROM replicas r JOIN deployments d ON d.id = r.deployment_id
			JOIN nodes n ON n.id = r.node_id WHERE d.status = 'active' AND r.desired = 'running' AND n.status = 'active'
			AND n.last_seen_at > now() - interval '90 seconds')
		SELECT m.id, m.name, m.display_name, m.route_policy, m.status,
			(SELECT count(*) FROM healthy h WHERE h.model_id = m.id AND h.state = 'ready'),
			(SELECT count(*) FROM healthy h WHERE h.model_id = m.id AND h.state NOT IN ('failed', 'lost', 'stopped')),
			(SELECT count(*) FROM healthy h WHERE h.model_id = m.id AND h.state IN ('assigned', 'staging', 'starting')),
			(SELECT count(DISTINCT h.node_id) FROM healthy h WHERE h.model_id = m.id AND h.state = 'ready'),
			coalesce((SELECT sum((h.live->>'tokens_per_second')::float8) FROM healthy h WHERE h.model_id = m.id AND h.live_at > now() - `+fresh+`), 0),
			(SELECT count(*) FROM admission_queue a WHERE a.model_id = m.id AND a.heartbeat_at > now() - interval '30 seconds' AND a.deadline > now()),
			(SELECT count(*) FROM provider_routes pr JOIN providers p ON p.id = pr.provider_id WHERE pr.model_id = m.id AND pr.enabled AND p.enabled),
			(SELECT count(*) FROM deployments d WHERE d.model_id = m.id AND d.status = 'active' AND d.mode <> 'warm')
		FROM models m ORDER BY m.name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []ModelLive{}
	for rows.Next() {
		var m ModelLive
		var status string
		var loading, onDemand int
		if err := rows.Scan(&m.ID, &m.Name, &m.DisplayName, &m.RoutePolicy, &status, &m.Ready, &m.Replicas, &loading, &m.Nodes,
			&m.TokensPerSecond, &m.Queued, &m.CloudRoutes, &onDemand); err != nil {
			return nil, err
		}
		m.Availability = availability(m, status, loading, onDemand, offline)
		out = append(out, m)
	}
	return out, rows.Err()
}

func availability(m ModelLive, status string, loading, onDemand int, offline bool) string {
	local := m.RoutePolicy != "cloud_only"
	cloud := m.RoutePolicy != "local_only" && !offline && m.CloudRoutes > 0
	switch {
	case status != "active":
		return "disabled"
	case local && m.Ready > 0:
		return "available"
	case cloud:
		return "cloud"
	case local && loading > 0:
		return "loading"
	case local && onDemand > 0:
		return "standby"
	}
	return "unavailable"
}

// Personal is the view for people without organisation roles.
type Personal struct {
	InFlight      int               `json:"in_flight"`
	RequestsToday int               `json:"requests_today"`
	TokensToday   int64             `json:"tokens_today"`
	Projects      []PersonalProject `json:"projects"`
}

// PersonalProject is a project the person can use and its models.
type PersonalProject struct {
	Slug   string   `json:"slug"`
	Name   string   `json:"name"`
	Models []string `json:"models"`
}

func personal(ctx context.Context, q db.Querier, userID string, projects []string) (Personal, error) {
	p := Personal{Projects: []PersonalProject{}}
	if err := q.QueryRow(ctx, `SELECT (SELECT count(*) FROM leases WHERE principal_type = 'user' AND principal_id = $1 AND heartbeat_at > now() - interval '60 seconds'),
		(SELECT count(*) FROM usage_events WHERE user_id = $1 AND created_at >= date_trunc('day', now())),
		coalesce((SELECT sum(prompt_tokens + completion_tokens) FROM usage_events WHERE user_id = $1 AND created_at >= date_trunc('day', now())), 0)::bigint`,
		userID).Scan(&p.InFlight, &p.RequestsToday, &p.TokensToday); err != nil {
		return p, err
	}
	rows, err := q.Query(ctx, `SELECT pr.slug, pr.name, coalesce(array_agg(DISTINCT m.name ORDER BY m.name) FILTER (WHERE m.name IS NOT NULL), '{}')
		FROM projects pr LEFT JOIN model_grants g ON g.project_id = pr.id OR g.project_id IS NULL
		LEFT JOIN models m ON m.id = g.model_id AND m.status = 'active'
		WHERE pr.id = ANY($1::uuid[]) AND pr.archived_at IS NULL GROUP BY pr.slug, pr.name ORDER BY pr.slug`, projects)
	if err != nil {
		return p, err
	}
	defer rows.Close()
	for rows.Next() {
		var pp PersonalProject
		if err := rows.Scan(&pp.Slug, &pp.Name, &pp.Models); err != nil {
			return p, err
		}
		p.Projects = append(p.Projects, pp)
	}
	return p, rows.Err()
}
