package fleet

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Scope decides which parts of the live view a viewer may see.
type Scope struct {
	Org        bool
	Usage      bool
	Nodes      bool
	Models     bool
	UserID     string
	ProjectIDs []string
}

// Snapshot is the live operations view; sections the viewer may not see are nil.
type Snapshot struct {
	At         time.Time   `json:"at"`
	Org        bool        `json:"org"`
	Tiles      *Tiles      `json:"tiles,omitempty"`
	Throughput []float64   `json:"throughput,omitempty"`
	Requests   []float64   `json:"requests,omitempty"`
	InFlight   []InFlight  `json:"in_flight"`
	Connected  []Connected `json:"connected"`
	Nodes      []NodeLive  `json:"nodes"`
	Models     []ModelLive `json:"models"`
	Personal   *Personal   `json:"personal,omitempty"`
}

// Tiles are the headline numbers.
type Tiles struct {
	TokensPerSecond       float64  `json:"tokens_per_second"`
	PromptTokensPerSecond float64  `json:"prompt_tokens_per_second"`
	InFlightLocal         int      `json:"in_flight_local"`
	InFlightCloud         int      `json:"in_flight_cloud"`
	Queued                int      `json:"queued"`
	ActiveUsers           int      `json:"active_users"`
	NodesOnline           int      `json:"nodes_online"`
	NodesTotal            int      `json:"nodes_total"`
	ModelsAvailable       int      `json:"models_available"`
	ModelsTotal           int      `json:"models_total"`
	TokensToday           int64    `json:"tokens_today"`
	CostTodayMicros       int64    `json:"cost_today_micros"`
	CostToday             string   `json:"cost_today"`
	CloudShareToday       float64  `json:"cloud_share_today"`
	Offline               bool     `json:"offline"`
	Warnings              []string `json:"warnings"`
}

// Rate formats a tokens-per-second figure.
func Rate(v float64) string {
	switch {
	case v >= 1000:
		return strconv.FormatFloat(v/1000, 'f', 1, 64) + "k"
	case v >= 100:
		return strconv.FormatFloat(v, 'f', 0, 64)
	}
	return strconv.FormatFloat(v, 'f', 1, 64)
}

// Percent formats a 0..1 share.
func Percent(v float64) string { return strconv.FormatFloat(v*100, 'f', 0, 64) + "%" }

// Load builds the live view for a viewer.
func Load(ctx context.Context, q db.Querier, s Scope) (Snapshot, error) {
	snap := Snapshot{At: time.Now().UTC(), Org: s.Org, InFlight: []InFlight{}, Connected: []Connected{}, Nodes: []NodeLive{}, Models: []ModelLive{}}
	if !s.Org {
		p, err := personal(ctx, q, s.UserID, s.ProjectIDs)
		snap.Personal = &p
		return snap, err
	}
	t, err := tiles(ctx, q)
	if err != nil {
		return snap, err
	}
	snap.Tiles = &t
	if snap.Throughput, snap.Requests, err = series(ctx, q, t.TokensPerSecond, snap.At); err != nil {
		return snap, err
	}
	if s.Usage {
		if snap.InFlight, err = inFlight(ctx, q); err != nil {
			return snap, err
		}
		if snap.Connected, err = connected(ctx, q); err != nil {
			return snap, err
		}
	}
	if s.Nodes {
		nodes, err := Nodes(ctx, q)
		if err != nil {
			return snap, err
		}
		for _, n := range nodes {
			snap.Nodes = append(snap.Nodes, n.Row())
		}
	}
	models, err := modelsLive(ctx, q, t.Offline)
	if err != nil {
		return snap, err
	}
	for _, m := range models {
		if m.Available() {
			snap.Tiles.ModelsAvailable++
		}
	}
	if s.Models {
		snap.Models = models
	}
	return snap, nil
}

const fresh = "interval '" + staleAfter + "'"

func tiles(ctx context.Context, q db.Querier) (Tiles, error) {
	var t Tiles
	var stale, failed, pending int
	var cloudToday, allToday int64
	err := q.QueryRow(ctx, `SELECT
		coalesce((SELECT sum((live->>'tokens_per_second')::float8) FROM replicas WHERE live_at > now() - `+fresh+`), 0),
		coalesce((SELECT sum((live->>'prompt_tokens_per_second')::float8) FROM replicas WHERE live_at > now() - `+fresh+`), 0),
		(SELECT count(*) FROM leases WHERE heartbeat_at > now() - interval '60 seconds' AND replica_id IS NOT NULL),
		(SELECT count(*) FROM leases WHERE heartbeat_at > now() - interval '60 seconds' AND route_id IS NOT NULL),
		(SELECT count(*) FROM admission_queue WHERE heartbeat_at > now() - interval '30 seconds' AND deadline > now()),
		(SELECT count(DISTINCT principal_id) FROM (
			SELECT principal_id FROM leases WHERE principal_type = 'user' AND heartbeat_at > now() - interval '60 seconds'
			UNION SELECT principal_id FROM usage_events WHERE principal_type = 'user' AND created_at > now() - interval '5 minutes') a),
		(SELECT count(*) FROM nodes WHERE status = 'active' AND last_seen_at > now() - interval '90 seconds'),
		(SELECT count(*) FROM nodes WHERE status <> 'revoked'),
		(SELECT count(*) FROM nodes WHERE status = 'active' AND (last_seen_at IS NULL OR last_seen_at <= now() - interval '90 seconds')),
		(SELECT count(*) FROM replicas WHERE desired = 'running' AND state = 'failed'),
		(SELECT count(*) FROM nodes WHERE status = 'pending'),
		(SELECT count(*) FROM models WHERE status = 'active'),
		coalesce((SELECT sum(prompt_tokens + completion_tokens) FROM usage_events WHERE created_at >= date_trunc('day', now())), 0)::bigint,
		coalesce((SELECT sum(cost_micros) FROM usage_events WHERE created_at >= date_trunc('day', now())), 0)::bigint,
		(SELECT count(*) FROM usage_events WHERE created_at >= date_trunc('day', now()) AND route_kind = 'provider'),
		(SELECT count(*) FROM usage_events WHERE created_at >= date_trunc('day', now())),
		coalesce((SELECT offline FROM settings WHERE id), true)`).
		Scan(&t.TokensPerSecond, &t.PromptTokensPerSecond, &t.InFlightLocal, &t.InFlightCloud, &t.Queued, &t.ActiveUsers, &t.NodesOnline,
			&t.NodesTotal, &stale, &failed, &pending, &t.ModelsTotal, &t.TokensToday, &t.CostTodayMicros, &cloudToday, &allToday, &t.Offline)
	if err != nil {
		return t, err
	}
	t.CostToday = Dollars(t.CostTodayMicros)
	if allToday > 0 {
		t.CloudShareToday = float64(cloudToday) / float64(allToday)
	}
	t.Warnings = []string{}
	if stale > 0 {
		t.Warnings = append(t.Warnings, plural(stale, "node has", "nodes have")+" not checked in for 90 seconds")
	}
	if failed > 0 {
		t.Warnings = append(t.Warnings, plural(failed, "replica has", "replicas have")+" failed to start")
	}
	if pending > 0 {
		t.Warnings = append(t.Warnings, plural(pending, "node is", "nodes are")+" waiting for approval")
	}
	if t.Offline {
		t.Warnings = append(t.Warnings, "Offline mode is on: cloud providers are disabled")
	}
	return t, nil
}

func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return strconv.Itoa(n) + " " + many
}

// series is completed tokens per second and requests for each of the last 60 minutes, then the live rate.
func series(ctx context.Context, q db.Querier, live float64, now time.Time) ([]float64, []float64, error) {
	start := now.Truncate(time.Minute).Add(-59 * time.Minute)
	tokens, reqs := make([]float64, 60), make([]float64, 60)
	rows, err := q.Query(ctx, `SELECT date_trunc('minute', created_at), sum(completion_tokens)::bigint, count(*) FROM usage_events
		WHERE created_at >= $1 GROUP BY 1`, start)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var m time.Time
		var tok, n int64
		if err := rows.Scan(&m, &tok, &n); err != nil {
			return nil, nil, err
		}
		i := int(m.UTC().Sub(start) / time.Minute)
		if i < 0 || i >= 60 {
			continue
		}
		span := 60.0
		if i == 59 {
			span = max(1, now.Sub(now.Truncate(time.Minute)).Seconds())
		}
		tokens[i], reqs[i] = float64(tok)/span, float64(n)
	}
	return append(tokens, live), reqs, rows.Err()
}

// Spark draws values as an SVG polyline in a width by height box, scaled to the largest value.
func Spark(values []float64, w, h float64) string {
	if len(values) == 0 {
		return ""
	}
	top := 0.0
	for _, v := range values {
		top = max(top, v)
	}
	var b strings.Builder
	step := w / float64(max(1, len(values)-1))
	for i, v := range values {
		y := h - 1
		if top > 0 {
			y = h - 1 - (v/top)*(h-2)
		}
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(strconv.FormatFloat(float64(i)*step, 'f', 1, 64) + "," + strconv.FormatFloat(y, 'f', 1, 64))
	}
	return b.String()
}

// Area closes a sparkline into a filled shape along the bottom edge.
func Area(values []float64, w, h float64) string {
	line := Spark(values, w, h)
	if line == "" {
		return ""
	}
	return "0," + strconv.FormatFloat(h, 'f', 1, 64) + " " + line + " " + strconv.FormatFloat(w, 'f', 1, 64) + "," + strconv.FormatFloat(h, 'f', 1, 64)
}
