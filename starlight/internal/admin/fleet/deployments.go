package fleet

import (
	"bytes"
	"context"
	"encoding/json"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Deployment is a model revision plus where and how to run it.
type Deployment struct {
	ID          string
	Name        string
	ModelID     string
	ModelName   string
	Revision    int
	Spec        wire.Spec
	Pool        string
	Selector    map[string]string
	Mode        string
	MinReplicas int
	MaxReplicas int
	IdleSeconds int
	Status      string
	Ready       int
	Live        int
	UpdatedAt   time.Time
}

// DeploymentInput is what the deployment form or configuration file sets.
type DeploymentInput struct {
	Name        string
	ModelID     string
	Spec        wire.Spec
	Pool        string
	Selector    map[string]string
	Mode        string
	MinReplicas int
	MaxReplicas int
	IdleSeconds int
}

func (in *DeploymentInput) validate() error {
	if in.Pool == "" {
		in.Pool = "default"
	}
	if in.Mode == "" {
		in.Mode = "warm"
	}
	if in.IdleSeconds == 0 {
		in.IdleSeconds = 900
	}
	if in.MaxReplicas == 0 {
		in.MaxReplicas = max(1, in.MinReplicas)
	}
	if in.Selector == nil {
		in.Selector = map[string]string{}
	}
	if err := slug("deployment name", in.Name); err != nil {
		return err
	}
	if !labelPattern.MatchString(in.Pool) {
		return Invalid("pool must be letters, digits, dots, dashes or underscores")
	}
	if err := oneOf("mode", in.Mode, "warm", "on_demand", "batch"); err != nil {
		return err
	}
	if in.MinReplicas < 0 || in.MaxReplicas < 1 || in.MinReplicas > in.MaxReplicas || in.MaxReplicas > 256 {
		return Invalid("replicas must satisfy 0 <= minimum <= maximum <= 256, with a maximum of at least 1")
	}
	if in.IdleSeconds < 60 || in.IdleSeconds > 30*24*3600 {
		return Invalid("idle time must be between 60 seconds and 30 days")
	}
	if err := labels(in.Selector); err != nil {
		return err
	}
	if err := in.Spec.Validate(); err != nil {
		return Invalid("engine spec: " + err.Error())
	}
	return nil
}

const deploymentCols = `d.id, d.name, d.model_id, m.name, d.revision, d.spec, d.pool, d.selector, d.mode, d.min_replicas, d.max_replicas,
	d.idle_seconds, d.status, d.updated_at,
	(SELECT count(*) FROM replicas r WHERE r.deployment_id = d.id AND r.state = 'ready' AND r.desired = 'running'),
	(SELECT count(*) FROM replicas r WHERE r.deployment_id = d.id AND r.desired = 'running' AND r.state NOT IN ('failed', 'lost', 'stopped'))`

func scanDeployment(row interface{ Scan(...any) error }) (Deployment, error) {
	var d Deployment
	var spec, sel []byte
	err := row.Scan(&d.ID, &d.Name, &d.ModelID, &d.ModelName, &d.Revision, &spec, &d.Pool, &sel, &d.Mode, &d.MinReplicas, &d.MaxReplicas,
		&d.IdleSeconds, &d.Status, &d.UpdatedAt, &d.Ready, &d.Live)
	if err != nil {
		return d, notFound(err)
	}
	_ = json.Unmarshal(spec, &d.Spec)
	_ = json.Unmarshal(sel, &d.Selector)
	return d, nil
}

// Deployments lists deployments of a model, or all when modelID is empty.
func Deployments(ctx context.Context, q db.Querier, modelID string) ([]Deployment, error) {
	rows, err := q.Query(ctx, "SELECT "+deploymentCols+` FROM deployments d JOIN models m ON m.id = d.model_id
		WHERE $1 = '' OR d.model_id::text = $1 ORDER BY d.status, d.name`, modelID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Deployment
	for rows.Next() {
		d, err := scanDeployment(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// DeploymentByName loads one deployment.
func DeploymentByName(ctx context.Context, q db.Querier, name string) (Deployment, error) {
	return scanDeployment(q.QueryRow(ctx, "SELECT "+deploymentCols+" FROM deployments d JOIN models m ON m.id = d.model_id WHERE d.name = $1", name))
}

// CreateDeployment adds a deployment at revision 1.
func (t *Tx) CreateDeployment(ctx context.Context, in DeploymentInput) (Deployment, error) {
	if err := in.validate(); err != nil {
		return Deployment{}, err
	}
	spec, _ := json.Marshal(in.Spec)
	sel, _ := json.Marshal(in.Selector)
	id := ids.New()
	_, err := t.tx.Exec(ctx, `INSERT INTO deployments (id, model_id, name, spec, pool, selector, mode, min_replicas, max_replicas, idle_seconds)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`, id, in.ModelID, in.Name, spec, in.Pool, sel, in.Mode, in.MinReplicas, in.MaxReplicas, in.IdleSeconds)
	if db.IsForeignKey(err) {
		return Deployment{}, ErrNotFound
	}
	if err != nil {
		return Deployment{}, uniqueMsg(err, "a deployment with that name already exists")
	}
	if err := t.record(ctx, "deployment.created", "deployment", id, map[string]any{"name": in.Name, "model_id": in.ModelID,
		"artifact": in.Spec.Model.Key(), "mode": in.Mode}); err != nil {
		return Deployment{}, err
	}
	return DeploymentByName(ctx, t.tx, in.Name)
}

// UpdateDeployment changes placement and scaling; a changed spec becomes a new revision rolled out surge-first.
func (t *Tx) UpdateDeployment(ctx context.Context, id string, in DeploymentInput) (Deployment, error) {
	if err := in.validate(); err != nil {
		return Deployment{}, err
	}
	var oldSpec []byte
	var name string
	if err := t.tx.QueryRow(ctx, "SELECT spec, name FROM deployments WHERE id = $1 FOR UPDATE", id).Scan(&oldSpec, &name); err != nil {
		return Deployment{}, notFound(err)
	}
	spec, _ := json.Marshal(in.Spec)
	sel, _ := json.Marshal(in.Selector)
	var normalized wire.Spec
	_ = json.Unmarshal(oldSpec, &normalized)
	old, _ := json.Marshal(normalized)
	bump := !bytes.Equal(old, spec)
	if _, err := t.tx.Exec(ctx, `UPDATE deployments SET spec = $2, pool = $3, selector = $4, mode = $5, min_replicas = $6, max_replicas = $7,
		idle_seconds = $8, revision = revision + CASE WHEN $9 THEN 1 ELSE 0 END, updated_at = now() WHERE id = $1`,
		id, spec, in.Pool, sel, in.Mode, in.MinReplicas, in.MaxReplicas, in.IdleSeconds, bump); err != nil {
		return Deployment{}, err
	}
	if err := t.record(ctx, "deployment.updated", "deployment", id, map[string]any{"new_revision": bump, "artifact": in.Spec.Model.Key(),
		"mode": in.Mode, "min": in.MinReplicas, "max": in.MaxReplicas}); err != nil {
		return Deployment{}, err
	}
	return DeploymentByName(ctx, t.tx, name)
}

// SetDeploymentStatus pauses, resumes or retires a deployment; paused and retired ones run no replicas.
func (t *Tx) SetDeploymentStatus(ctx context.Context, id, status string) error {
	if err := oneOf("status", status, "active", "paused", "retired"); err != nil {
		return err
	}
	tag, err := t.tx.Exec(ctx, "UPDATE deployments SET status = $2, updated_at = now() WHERE id = $1", id, status)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	if status != "active" {
		if err := t.stopReplicas(ctx, "deployment_id = $1", id); err != nil {
			return err
		}
	}
	return t.record(ctx, "deployment."+status, "deployment", id, nil)
}

// stopReplicas marks matching running replicas stopped and wakes every node they run on, rank group members included.
func (t *Tx) stopReplicas(ctx context.Context, where string, arg any) error {
	rows, err := t.tx.Query(ctx, `WITH stopped AS (UPDATE replicas SET desired = 'stopped', generation = generation + 1, updated_at = now()
		WHERE desired = 'running' AND `+where+` RETURNING id, node_id)
		SELECT node_id FROM stopped UNION SELECT m.node_id FROM replica_members m JOIN stopped s ON s.id = m.replica_id`, arg)
	if err != nil {
		return err
	}
	var nodes []string
	for rows.Next() {
		var n string
		if err := rows.Scan(&n); err != nil {
			rows.Close()
			return err
		}
		nodes = append(nodes, n)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	for _, n := range nodes {
		if err := t.wakeNode(ctx, n); err != nil {
			return err
		}
	}
	return nil
}

// Replica is one placed copy of a deployment.
type Replica struct {
	ID          string
	Deployment  string
	NodeName    string
	Revision    int
	Desired     string
	State       string
	Error       string
	Slots       int
	ReadyAt     *time.Time
	LastRequest *time.Time
}

// Replicas lists replicas of a deployment or on a node; one of the two filters must be set.
func Replicas(ctx context.Context, q db.Querier, deploymentID, nodeID string) ([]Replica, error) {
	rows, err := q.Query(ctx, `SELECT r.id, d.name, n.name, r.deployment_revision, r.desired, r.state, r.error, r.slots, r.ready_at, r.last_request_at
		FROM replicas r JOIN deployments d ON d.id = r.deployment_id JOIN nodes n ON n.id = r.node_id
		WHERE ($1 <> '' AND r.deployment_id::text = $1) OR ($2 <> '' AND (r.node_id::text = $2
			OR EXISTS (SELECT 1 FROM replica_members m WHERE m.replica_id = r.id AND m.node_id::text = $2))) ORDER BY d.name, n.name`, deploymentID, nodeID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Replica
	for rows.Next() {
		var r Replica
		if err := rows.Scan(&r.ID, &r.Deployment, &r.NodeName, &r.Revision, &r.Desired, &r.State, &r.Error, &r.Slots, &r.ReadyAt, &r.LastRequest); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
