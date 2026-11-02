package scheduler

import (
	"context"
	"encoding/json"
	"net"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

type deployment struct {
	id, modelID, pool, mode string
	revision                int
	spec                    wire.Spec
	rawSpec                 []byte
	selector                map[string]string
	min, max, idle          int
}

type node struct {
	id       string
	pool     string
	labels   map[string]string
	backend  string
	rankHost string
	memory   uint64
	reserved uint64
	cached   map[string]bool
	gpus     []wire.GPU
	taken    map[string]bool
	ports    map[int]bool
}

// held is one node's part of a replica, the lead's or a follower member's, and the GPUs it pins there.
type held struct {
	node, state string
	devices     []string
}

type replica struct {
	id, deployment, node, desired, state string
	revision                             int
	readyAt, lastRequest                 *time.Time
	updated                              time.Time
	devices                              []string
	members                              []held
	share                                uint64
	port                                 int
}

func (r replica) live() bool {
	return r.desired == "running" && r.state != "failed" && r.state != "lost" && r.state != "stopped"
}

// seats lists every node the replica occupies, its lead first.
func (r replica) seats() []held {
	return append([]held{{node: r.node, state: r.state, devices: r.devices}}, r.members...)
}

// holds reports whether a seat still holds its GPUs: until the replica is told to stop and its process has.
func (r replica) holds(h held) bool {
	return r.desired == "running" || h.state != "stopped" && h.state != "failed" && h.state != "lost"
}

// snapshot is what one reconcile pass decides from.
type snapshot struct {
	deployments []deployment
	nodes       []*node
	replicas    map[string][]replica
	failedOn    map[string]map[string]bool
	queue       map[string]int
}

func load(ctx context.Context, q db.Querier, queue map[string]int) (*snapshot, error) {
	s := &snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, queue: queue}
	rows, err := q.Query(ctx, `SELECT id, model_id, pool, mode, revision, spec, selector, min_replicas, max_replicas, idle_seconds
		FROM deployments WHERE status = 'active' ORDER BY id`)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var d deployment
		var sel []byte
		if err := rows.Scan(&d.id, &d.modelID, &d.pool, &d.mode, &d.revision, &d.rawSpec, &sel, &d.min, &d.max, &d.idle); err != nil {
			rows.Close()
			return nil, err
		}
		_ = json.Unmarshal(d.rawSpec, &d.spec)
		_ = json.Unmarshal(sel, &d.selector)
		s.deployments = append(s.deployments, d)
	}
	rows.Close()
	if err := loadNodes(ctx, q, s); err != nil {
		return nil, err
	}
	if err := loadReplicas(ctx, q, s); err != nil {
		return nil, err
	}
	s.account()
	return s, nil
}

func loadNodes(ctx context.Context, q db.Querier, s *snapshot) error {
	rows, err := q.Query(ctx, `SELECT n.id, n.pool, n.labels, coalesce(n.facts->>'backend', ''), coalesce((n.facts->>'memory_bytes')::numeric, 0)::bigint,
		n.cached_artifacts, coalesce(n.facts->'gpus', '[]'::jsonb), coalesce(n.facts->>'rank_host', ''), n.address
		FROM nodes n WHERE n.status = 'active' AND n.last_seen_at > now() - interval '90 seconds' ORDER BY n.id`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		n := &node{cached: map[string]bool{}, taken: map[string]bool{}, ports: map[int]bool{}}
		var labels, gpus []byte
		var mem int64
		var cached []string
		var address string
		if err := rows.Scan(&n.id, &n.pool, &labels, &n.backend, &mem, &cached, &gpus, &n.rankHost, &address); err != nil {
			return err
		}
		_ = json.Unmarshal(labels, &n.labels)
		_ = json.Unmarshal(gpus, &n.gpus)
		n.memory = uint64(max(mem, 0))
		if n.rankHost == "" {
			n.rankHost, _, _ = net.SplitHostPort(address)
		}
		for _, c := range cached {
			n.cached[c] = true
		}
		s.nodes = append(s.nodes, n)
	}
	return rows.Err()
}

func loadReplicas(ctx context.Context, q db.Querier, s *snapshot) error {
	rows, err := q.Query(ctx, `SELECT id, deployment_id, node_id, desired, state, deployment_revision, ready_at, last_request_at, updated_at, devices,
		coalesce((spec->>'memory_gib')::bigint, 0), greatest(coalesce((spec->>'nodes')::int, 1), 1), coalesce(rank_port, 0) FROM replicas ORDER BY id`)
	if err != nil {
		return err
	}
	where := map[string]int{}
	for rows.Next() {
		var r replica
		var gib, span int64
		if err := rows.Scan(&r.id, &r.deployment, &r.node, &r.desired, &r.state, &r.revision, &r.readyAt, &r.lastRequest, &r.updated, &r.devices,
			&gib, &span, &r.port); err != nil {
			rows.Close()
			return err
		}
		r.share = divUp(uint64(max(gib, 0))<<30, uint64(span))
		where[r.id] = len(s.replicas[r.deployment])
		s.replicas[r.deployment] = append(s.replicas[r.deployment], r)
		if r.state == "failed" && time.Since(r.updated) < 10*time.Minute {
			s.failed(r.deployment, r.node)
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	return loadMembers(ctx, q, s, where)
}

func loadMembers(ctx context.Context, q db.Querier, s *snapshot, where map[string]int) error {
	rows, err := q.Query(ctx, `SELECT m.replica_id, r.deployment_id, m.node_id, m.state, m.devices, m.updated_at FROM replica_members m
		JOIN replicas r ON r.id = m.replica_id ORDER BY m.replica_id, m.member`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var id, dep string
		var h held
		var updated time.Time
		if err := rows.Scan(&id, &dep, &h.node, &h.state, &h.devices, &updated); err != nil {
			return err
		}
		if i, ok := where[id]; ok {
			s.replicas[dep][i].members = append(s.replicas[dep][i].members, h)
		}
		if h.state == "failed" && time.Since(updated) < 10*time.Minute {
			s.failed(dep, h.node)
		}
	}
	return rows.Err()
}

func (s *snapshot) failed(dep, node string) {
	if s.failedOn[dep] == nil {
		s.failedOn[dep] = map[string]bool{}
	}
	s.failedOn[dep][node] = true
}

// account charges every seat to its node until its process has stopped: its GPUs, its memory and a group's ports.
func (s *snapshot) account() {
	byID := map[string]*node{}
	for _, n := range s.nodes {
		byID[n.id] = n
	}
	for _, rs := range s.replicas {
		for _, r := range rs {
			for i, h := range r.seats() {
				n := byID[h.node]
				if n == nil || !r.holds(h) {
					continue
				}
				for _, d := range h.devices {
					n.taken[d] = true
				}
				n.reserved += r.share
				if i == 0 && r.port > 0 {
					n.ports[r.port] = true
				}
			}
		}
	}
}

// draining reports a replica told to stop with a seat whose process has not stopped yet.
func (r replica) draining() bool {
	if r.desired == "running" {
		return false
	}
	for _, h := range r.seats() {
		if r.holds(h) {
			return true
		}
	}
	return false
}

func divUp(a, b uint64) uint64 {
	return (a + b - 1) / max(b, 1)
}
