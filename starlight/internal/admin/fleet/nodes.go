package fleet

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Live is a replica's or node's current load from fresh engine stats.
type Live struct {
	TokensPerSecond       float64 `json:"tokens_per_second"`
	PromptTokensPerSecond float64 `json:"prompt_tokens_per_second"`
	Running               int     `json:"running"`
	Waiting               int     `json:"waiting"`
	KVUsage               float64 `json:"kv_usage"`
	Fresh                 bool    `json:"fresh"`
}

// staleAfter is how old a replica's live stats may be before dashboards ignore them.
const staleAfter = "15 seconds"

// Node is an enrolled machine with its facts and current load.
type Node struct {
	ID           string
	Name         string
	Status       string
	Pool         string
	Labels       map[string]string
	MaxClass     string
	Address      string
	Facts        wire.Facts
	AgentVersion string
	LastSeen     *time.Time
	CertNotAfter *time.Time
	Ready        int
	Replicas     int
	Live         Live
}

// State is the node's operational state as an operator sees it: online, stale, draining, disabled, pending or revoked.
func (n Node) State() string {
	if n.Status == "active" {
		if n.LastSeen != nil && time.Since(*n.LastSeen) <= 90*time.Second {
			return "online"
		}
		return "stale"
	}
	return n.Status
}

// Accelerators names the chip or GPUs.
func (n Node) Accelerators() string {
	var out []string
	for _, g := range n.Facts.GPUs {
		out = append(out, g.Name)
	}
	return strings.Join(out, ", ")
}

const nodeCols = `n.id, n.name, n.status, n.pool, n.labels, n.max_classification, n.address, n.facts, n.agent_version, n.last_seen_at, n.cert_not_after,
	(SELECT count(*) FROM replicas r WHERE r.node_id = n.id AND r.state = 'ready' AND r.desired = 'running') + (SELECT count(*) FROM replica_members m
		JOIN replicas r ON r.id = m.replica_id WHERE m.node_id = n.id AND m.state = 'ready' AND r.desired = 'running'),
	(SELECT count(*) FROM replicas r WHERE r.node_id = n.id AND r.desired = 'running') + (SELECT count(*) FROM replica_members m
		JOIN replicas r ON r.id = m.replica_id WHERE m.node_id = n.id AND r.desired = 'running'),
	coalesce((SELECT sum((r.live->>'tokens_per_second')::float8) FROM replicas r WHERE r.node_id = n.id AND r.live_at > now() - interval '` + staleAfter + `'), 0),
	coalesce((SELECT sum((r.live->>'prompt_tokens_per_second')::float8) FROM replicas r WHERE r.node_id = n.id AND r.live_at > now() - interval '` + staleAfter + `'), 0),
	coalesce((SELECT sum((r.live->>'running')::int) FROM replicas r WHERE r.node_id = n.id AND r.live_at > now() - interval '` + staleAfter + `'), 0),
	coalesce((SELECT sum((r.live->>'waiting')::int) FROM replicas r WHERE r.node_id = n.id AND r.live_at > now() - interval '` + staleAfter + `'), 0),
	coalesce((SELECT max((r.live->>'kv_usage')::float8) FROM replicas r WHERE r.node_id = n.id AND r.live_at > now() - interval '` + staleAfter + `'), 0),
	EXISTS (SELECT 1 FROM replicas r WHERE r.node_id = n.id AND r.live_at > now() - interval '` + staleAfter + `')`

func scanNode(row interface{ Scan(...any) error }) (Node, error) {
	var n Node
	var lbl, facts []byte
	err := row.Scan(&n.ID, &n.Name, &n.Status, &n.Pool, &lbl, &n.MaxClass, &n.Address, &facts, &n.AgentVersion, &n.LastSeen, &n.CertNotAfter,
		&n.Ready, &n.Replicas, &n.Live.TokensPerSecond, &n.Live.PromptTokensPerSecond, &n.Live.Running, &n.Live.Waiting, &n.Live.KVUsage, &n.Live.Fresh)
	if err != nil {
		return n, notFound(err)
	}
	_ = json.Unmarshal(lbl, &n.Labels)
	_ = json.Unmarshal(facts, &n.Facts)
	return n, nil
}

// Nodes lists every node, revoked ones last.
func Nodes(ctx context.Context, q db.Querier) ([]Node, error) {
	rows, err := q.Query(ctx, "SELECT "+nodeCols+" FROM nodes n ORDER BY n.status = 'revoked', n.name")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Node
	for rows.Next() {
		n, err := scanNode(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, rows.Err()
}

// NodeByName loads one node.
func NodeByName(ctx context.Context, q db.Querier, name string) (Node, error) {
	return scanNode(q.QueryRow(ctx, "SELECT "+nodeCols+" FROM nodes n WHERE n.name = $1", name))
}

// Node actions an operator may take.
const (
	NodeApprove = "approve"
	NodeOnline  = "online"
	NodeDrain   = "drain"
	NodeDisable = "disable"
	NodeRevoke  = "revoke"
)

var nodeTarget = map[string]string{NodeApprove: "active", NodeOnline: "active", NodeDrain: "draining", NodeDisable: "disabled", NodeRevoke: "revoked"}

// SetNodeStatus applies an operator action; revoking withdraws certificates and replicas, draining moves work away.
func (t *Tx) SetNodeStatus(ctx context.Context, name, action string) (string, error) {
	status, ok := nodeTarget[action]
	if !ok {
		return "", Invalid("unknown node action " + action)
	}
	var id, old string
	err := t.tx.QueryRow(ctx, "SELECT id, status FROM nodes WHERE name = $1 FOR UPDATE", name).Scan(&id, &old)
	if err != nil {
		return "", notFound(err)
	}
	if old == "revoked" {
		return "", Invalid("a revoked node cannot change state; enrol it again")
	}
	if action == NodeApprove && old != "pending" {
		return "", Invalid("only pending nodes need approval")
	}
	if _, err := t.tx.Exec(ctx, `UPDATE nodes SET status = $2, updated_at = now(), sync_generation = sync_generation + 1,
		approved_at = CASE WHEN $2 = 'active' AND approved_at IS NULL THEN now() ELSE approved_at END,
		approved_by = CASE WHEN $2 = 'active' AND approved_by IS NULL THEN $3::uuid ELSE approved_by END WHERE id = $1`,
		id, status, optID(t.actor.ActorID)); err != nil {
		return "", err
	}
	switch status {
	case "revoked":
		if _, err := t.tx.Exec(ctx, "UPDATE node_certificates SET revoked_at = now() WHERE node_id = $1 AND revoked_at IS NULL", id); err != nil {
			return "", err
		}
		if err := t.stopReplicas(ctx, "node_id = $1", id); err != nil {
			return "", err
		}
		if _, err := t.tx.Exec(ctx, "UPDATE replicas SET state = 'lost', updated_at = now() WHERE node_id = $1", id); err != nil {
			return "", err
		}
	case "draining":
		if err := t.stopReplicas(ctx, "(node_id = $1 OR id IN (SELECT replica_id FROM replica_members WHERE node_id = $1))", id); err != nil {
			return "", err
		}
	}
	if err := t.wakeNode(ctx, id); err != nil {
		return "", err
	}
	return id, t.record(ctx, "node."+action, "node", id, map[string]any{"name": name, "from": old, "to": status})
}

// UpdateNode changes where a node fits: its pool, labels and the highest data classification it may process.
func (t *Tx) UpdateNode(ctx context.Context, name, pool string, lbl map[string]string, maxClass string) error {
	if !labelPattern.MatchString(pool) {
		return Invalid("pool must be letters, digits, dots, dashes or underscores")
	}
	if err := labels(lbl); err != nil {
		return err
	}
	if err := class(maxClass); err != nil {
		return err
	}
	raw, _ := json.Marshal(lbl)
	var id string
	err := t.tx.QueryRow(ctx, "UPDATE nodes SET pool = $2, labels = $3, max_classification = $4, updated_at = now() WHERE name = $1 RETURNING id",
		name, pool, raw, maxClass).Scan(&id)
	if err != nil {
		return notFound(err)
	}
	return t.record(ctx, "node.updated", "node", id, map[string]any{"name": name, "pool": pool, "labels": lbl, "max_classification": maxClass})
}

func (t *Tx) wakeNode(ctx context.Context, id string) error {
	if _, err := t.tx.Exec(ctx, "UPDATE nodes SET sync_generation = sync_generation + 1 WHERE id = $1", id); err != nil {
		return err
	}
	_, err := t.tx.Exec(ctx, "SELECT pg_notify('starlight_node', $1)", id)
	return err
}

// EnrolmentToken is a stored enrolment token without its secret.
type EnrolmentToken struct {
	ID          string
	Pool        string
	Labels      map[string]string
	MaxClass    string
	AutoApprove bool
	Uses        int
	MaxUses     int
	ExpiresAt   time.Time
	RevokedAt   *time.Time
	CreatedAt   time.Time
}

// Usable reports whether the token can still enrol a node.
func (e EnrolmentToken) Usable() bool {
	return e.RevokedAt == nil && e.Uses < e.MaxUses && e.ExpiresAt.After(time.Now())
}

// EnrolmentTokens lists recent tokens.
func EnrolmentTokens(ctx context.Context, q db.Querier) ([]EnrolmentToken, error) {
	rows, err := q.Query(ctx, `SELECT id, pool, labels, max_classification, auto_approve, uses, max_uses, expires_at, revoked_at, created_at
		FROM enrollment_tokens ORDER BY created_at DESC LIMIT 100`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []EnrolmentToken
	for rows.Next() {
		var e EnrolmentToken
		var lbl []byte
		if err := rows.Scan(&e.ID, &e.Pool, &lbl, &e.MaxClass, &e.AutoApprove, &e.Uses, &e.MaxUses, &e.ExpiresAt, &e.RevokedAt, &e.CreatedAt); err != nil {
			return nil, err
		}
		_ = json.Unmarshal(lbl, &e.Labels)
		out = append(out, e)
	}
	return out, rows.Err()
}

// CreateEnrolmentToken mints a token; the secret is returned once.
func (t *Tx) CreateEnrolmentToken(ctx context.Context, s control.TokenSpec) (string, error) {
	if s.Pool == "" {
		s.Pool = "default"
	}
	if s.MaxClass == "" {
		s.MaxClass = "confidential"
	}
	if !labelPattern.MatchString(s.Pool) {
		return "", Invalid("pool must be letters, digits, dots, dashes or underscores")
	}
	if err := labels(s.Labels); err != nil {
		return "", err
	}
	if err := class(s.MaxClass); err != nil {
		return "", err
	}
	if s.MaxUses < 0 || s.MaxUses > 1000 || s.TTL < 0 || s.TTL > 30*24*time.Hour {
		return "", Invalid("a token allows at most 1000 enrolments and lives at most 30 days")
	}
	s.CreatedBy = t.actor.ActorID
	tok, id, err := control.CreateToken(ctx, t.tx, s)
	if err != nil {
		return "", err
	}
	return tok, t.record(ctx, "node.token_created", "enrollment_token", id, map[string]any{"pool": s.Pool, "labels": s.Labels,
		"auto_approve": s.AutoApprove, "max_uses": s.MaxUses, "max_classification": s.MaxClass})
}

// RevokeEnrolmentToken stops a token from enrolling more nodes.
func (t *Tx) RevokeEnrolmentToken(ctx context.Context, id string) error {
	tag, err := t.tx.Exec(ctx, "UPDATE enrollment_tokens SET revoked_at = now() WHERE id = $1 AND revoked_at IS NULL", id)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return t.record(ctx, "node.token_revoked", "enrollment_token", id, nil)
}
