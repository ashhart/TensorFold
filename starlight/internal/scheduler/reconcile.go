package scheduler

import (
	"context"
	"log/slog"
	"sort"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Controller reconciles desired deployments into replicas on nodes.
type Controller struct {
	Pool     *pgxpool.Pool
	Instance string
	Log      *slog.Logger
	Every    time.Duration
	Hourly   []func(context.Context) error
}

// action is one change the controller makes.
type action struct {
	start *startAction
	stop  string
}

type startAction struct {
	d     deployment
	seats []seat
	port  int
}

// Lead runs reconcile passes and housekeeping until ctx ends or leadership is lost.
func (c *Controller) Lead(ctx context.Context, epoch int64) {
	every := c.Every
	if every <= 0 {
		every = 5 * time.Second
	}
	demand := make(chan struct{}, 1)
	go c.listenDemand(ctx, demand)
	tick := time.NewTicker(every)
	defer tick.Stop()
	house := newHousekeeper(c.Pool, c.Hourly)
	for ctx.Err() == nil {
		if err := c.Pass(ctx, epoch); err != nil && ctx.Err() == nil && c.Log != nil {
			c.Log.Warn("reconcile failed", "error", err)
		}
		house.run(ctx)
		select {
		case <-ctx.Done():
		case <-tick.C:
		case <-demand:
		}
	}
}

func (c *Controller) listenDemand(ctx context.Context, out chan<- struct{}) {
	for ctx.Err() == nil {
		conn, err := c.Pool.Acquire(ctx)
		if err != nil {
			sleep(ctx, 2*time.Second)
			continue
		}
		if _, err := conn.Exec(ctx, "LISTEN starlight_demand"); err == nil {
			for {
				if _, err := conn.Conn().WaitForNotification(ctx); err != nil {
					break
				}
				select {
				case out <- struct{}{}:
				default:
				}
			}
		}
		conn.Conn().Close(context.Background())
		conn.Release()
		sleep(ctx, time.Second)
	}
}

// Pass computes and applies one round of changes under the fencing epoch.
func (c *Controller) Pass(ctx context.Context, epoch int64) error {
	queue, err := admission.QueueDepth(ctx, c.Pool)
	if err != nil {
		return err
	}
	return db.Tx(ctx, c.Pool, func(tx pgx.Tx) error {
		if err := Fence(ctx, tx, c.Instance, epoch); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE replicas SET state = 'lost', updated_at = now() WHERE state NOT IN ('lost', 'stopped')
			AND node_id IN (SELECT id FROM nodes WHERE last_seen_at < now() - interval '90 seconds' OR status IN ('disabled', 'revoked'))`); err != nil {
			return err
		}
		touched := map[string]bool{}
		if err := breakGroups(ctx, tx, epoch, touched); err != nil {
			return err
		}
		snap, err := load(ctx, tx, queue)
		if err != nil {
			return err
		}
		for _, d := range snap.deployments {
			for _, a := range plan(snap, d, time.Now()) {
				nodes, err := apply(ctx, tx, epoch, a)
				if err != nil {
					return err
				}
				for _, n := range nodes {
					touched[n] = true
				}
			}
		}
		if err := cleanup(ctx, tx, touched); err != nil {
			return err
		}
		for n := range touched {
			if _, err := tx.Exec(ctx, "UPDATE nodes SET sync_generation = sync_generation + 1 WHERE id = $1", n); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, "SELECT pg_notify('starlight_node', $1)", n); err != nil {
				return err
			}
		}
		return nil
	})
}

// plan decides at most one start and one stop per deployment per pass, so changes roll gradually.
func plan(s *snapshot, d deployment, now time.Time) []action {
	var current, old, ready []replica
	for _, r := range s.replicas[d.id] {
		if !r.live() {
			continue
		}
		if r.revision == d.revision {
			current = append(current, r)
			if r.state == "ready" {
				ready = append(ready, r)
			}
		} else {
			old = append(old, r)
		}
	}
	target := d.min
	if d.mode == "warm" {
		target = max(target, 1)
	}
	if s.queue[d.modelID] > 0 && d.mode != "warm" {
		target = max(target, 1)
		if len(ready) > 0 && len(current) == len(ready) && s.queue[d.modelID] > 2*len(ready) {
			target = max(target, len(current)+1)
		}
	}
	target = min(target, d.max)
	var acts []action
	if len(current) < target {
		if seats := s.place(d); seats != nil {
			port := 0
			if d.spec.Span() > 1 {
				port = seats[0].node.freePort()
			}
			if d.spec.Span() == 1 || port > 0 {
				acts = append(acts, action{start: &startAction{d: d, seats: seats, port: port}})
			}
		} else if !stopping(s.replicas[d.id]) && len(ready) == len(current) && !failing(s.replicas[d.id], d.revision) {
			for _, r := range old {
				if s.makesRoom(d, r) {
					return []action{{stop: r.id}}
				}
			}
		}
	}
	switch {
	case len(old) > 0 && len(ready) >= max(1, min(target, len(current))):
		acts = append(acts, action{stop: old[0].id})
	case len(current) > d.max || d.mode == "warm" && len(current) > target:
		acts = append(acts, action{stop: surplus(current).id})
	case d.mode != "warm" && s.queue[d.modelID] == 0 && len(current) > d.min:
		if r := idlest(current); r != nil && idle(*r, d.idle, now) {
			acts = append(acts, action{stop: r.id})
		}
	}
	return acts
}

// stopping reports a replica told to stop with a seat still held, so a full fleet frees one replica at a time.
func stopping(rs []replica) bool {
	for _, r := range rs {
		if r.draining() {
			return true
		}
	}
	return false
}

// failing reports a replica of the revision that failed, so a broken revision never stops the replicas still serving.
func failing(rs []replica, revision int) bool {
	for _, r := range rs {
		if r.revision == revision && r.state == "failed" {
			return true
		}
	}
	return false
}

func surplus(rs []replica) replica {
	sort.Slice(rs, func(i, j int) bool {
		if (rs[i].state == "ready") != (rs[j].state == "ready") {
			return rs[i].state != "ready"
		}
		return used(rs[i]).Before(used(rs[j]))
	})
	return rs[0]
}

func idlest(rs []replica) *replica {
	var best *replica
	for i := range rs {
		if rs[i].state != "ready" {
			continue
		}
		if best == nil || used(rs[i]).Before(used(*best)) {
			best = &rs[i]
		}
	}
	return best
}

func used(r replica) time.Time {
	if r.lastRequest != nil {
		return *r.lastRequest
	}
	if r.readyAt != nil {
		return *r.readyAt
	}
	return r.updated
}

// idle holds a replica for at least its residency window after it became ready and after its last request.
func idle(r replica, seconds int, now time.Time) bool {
	window := time.Duration(seconds) * time.Second
	return r.readyAt != nil && now.Sub(*r.readyAt) >= window && now.Sub(used(r)) >= window
}

// apply makes one change and returns every node it concerns: a group's lead and each member.
func apply(ctx context.Context, tx pgx.Tx, epoch int64, a action) ([]string, error) {
	if a.start != nil {
		lead, id := a.start.seats[0], ids.New()
		var port *int
		if a.start.port > 0 {
			port = &a.start.port
		}
		_, err := tx.Exec(ctx, `INSERT INTO replicas (id, deployment_id, deployment_revision, spec, node_id, desired, state, slots, epoch, devices, rank_port)
			VALUES ($1, $2, $3, $4, $5, 'running', 'assigned', $6, $7, $8, $9)`, id, a.start.d.id, a.start.d.revision, a.start.d.rawSpec,
			lead.node.id, max(a.start.d.spec.Parallel, 1), epoch, append([]string{}, lead.devices...), port)
		if err != nil {
			return nil, err
		}
		nodes := []string{lead.node.id}
		for i, m := range a.start.seats[1:] {
			if _, err := tx.Exec(ctx, "INSERT INTO replica_members (replica_id, member, node_id, devices) VALUES ($1, $2, $3, $4)",
				id, i+1, m.node.id, append([]string{}, m.devices...)); err != nil {
				return nil, err
			}
			nodes = append(nodes, m.node.id)
		}
		return nodes, nil
	}
	var node string
	if err := tx.QueryRow(ctx, `UPDATE replicas SET desired = 'stopped', generation = generation + 1, epoch = $2, updated_at = now()
		WHERE id = $1 RETURNING node_id`, a.stop, epoch).Scan(&node); err != nil {
		return nil, err
	}
	return append(memberNodes(ctx, tx, a.stop), node), nil
}

func memberNodes(ctx context.Context, tx pgx.Tx, replicaID string) []string {
	var nodes []string
	_ = tx.QueryRow(ctx, "SELECT coalesce(array_agg(node_id::text), '{}') FROM replica_members WHERE replica_id = $1", replicaID).Scan(&nodes)
	return nodes
}

// breakGroups stops a group whose lead or member was lost, or whose member failed: no rank serves without every rank.
func breakGroups(ctx context.Context, tx pgx.Tx, epoch int64, touched map[string]bool) error {
	if _, err := tx.Exec(ctx, `UPDATE replica_members SET state = 'lost', updated_at = now() WHERE state NOT IN ('lost', 'stopped')
		AND node_id IN (SELECT id FROM nodes WHERE last_seen_at < now() - interval '90 seconds' OR status IN ('disabled', 'revoked'))`); err != nil {
		return err
	}
	rows, err := tx.Query(ctx, `WITH broken AS (
			SELECT DISTINCT ON (r.id) r.id, CASE WHEN r.state = 'lost' THEN 'rank 0''s node was lost'
				ELSE 'rank member ' || m.member || ' ' || m.state || ': ' || m.error END AS why
			FROM replicas r JOIN replica_members m ON m.replica_id = r.id WHERE r.desired = 'running'
			AND (r.state = 'lost' OR m.state = 'lost' OR m.state = 'failed' AND m.updated_at < now() - interval '30 seconds') ORDER BY r.id, m.member)
		UPDATE replicas r SET desired = 'stopped', generation = r.generation + 1, epoch = $1, updated_at = now(), error = b.why
		FROM broken b WHERE b.id = r.id RETURNING r.id, r.node_id`, epoch)
	if err != nil {
		return err
	}
	var broken []string
	for rows.Next() {
		var id, node string
		if err := rows.Scan(&id, &node); err != nil {
			rows.Close()
			return err
		}
		broken, touched[node] = append(broken, id), true
	}
	rows.Close()
	for _, id := range broken {
		for _, n := range memberNodes(ctx, tx, id) {
			touched[n] = true
		}
	}
	return rows.Err()
}

// cleanup stops failed replicas so they are replaced elsewhere and deletes replicas that are finished.
func cleanup(ctx context.Context, tx pgx.Tx, touched map[string]bool) error {
	rows, err := tx.Query(ctx, `UPDATE replicas SET desired = 'stopped', generation = generation + 1, updated_at = now()
		WHERE desired = 'running' AND state = 'failed' AND updated_at < now() - interval '2 minutes' RETURNING id, node_id`)
	if err != nil {
		return err
	}
	var stopped []string
	for rows.Next() {
		var id, n string
		if err := rows.Scan(&id, &n); err != nil {
			rows.Close()
			return err
		}
		stopped, touched[n] = append(stopped, id), true
	}
	rows.Close()
	for _, id := range stopped {
		for _, n := range memberNodes(ctx, tx, id) {
			touched[n] = true
		}
	}
	_, err = tx.Exec(ctx, `DELETE FROM replicas WHERE (desired = 'stopped' AND state IN ('stopped', 'failed', 'lost') AND updated_at < now() - interval '5 minutes')
		OR (state = 'lost' AND updated_at < now() - interval '1 hour')`)
	return err
}
