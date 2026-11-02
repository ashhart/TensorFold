package control

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

var reportable = map[string]bool{wire.StateStaging: true, wire.StateStarting: true, wire.StateReady: true, wire.StateDraining: true,
	wire.StateStopped: true, wire.StateFailed: true}

func (s *Server) sync(w http.ResponseWriter, r *http.Request) {
	nodeID, cert, revoked, err := s.node(r)
	if err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	if revoked {
		httpx.JSON(w, http.StatusOK, wire.SyncResponse{NodeStatus: "revoked", Assignments: []wire.Assignment{}})
		return
	}
	var req wire.SyncRequest
	if err := decode(r, &req); err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	ctx := r.Context()
	if err := s.observe(ctx, nodeID, req); err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	resp, err := s.desired(ctx, nodeID)
	if err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	if resp.Generation == req.Generation {
		wake, cancel := s.subscribe(nodeID)
		poll := s.Poll
		if poll <= 0 {
			poll = 25 * time.Second
		}
		select {
		case <-wake:
		case <-time.After(poll):
		case <-ctx.Done():
		}
		cancel()
		if resp, err = s.desired(ctx, nodeID); err != nil {
			httpx.Write(w, httpx.Plain, err)
			return
		}
	}
	resp.RenewAfter = cert.NotBefore.Add(cert.NotAfter.Sub(cert.NotBefore) * 2 / 3).Unix()
	httpx.JSON(w, http.StatusOK, resp)
}

// observe stores what the node reports; replica states only apply to the generation the node was given.
func (s *Server) observe(ctx context.Context, nodeID string, req wire.SyncRequest) error {
	facts, _ := json.Marshal(req.Facts)
	cached := req.CachedArtifacts
	if cached == nil {
		cached = []string{}
	}
	if len(cached) > 512 {
		cached = cached[:512]
	}
	addr := req.Address
	if !validAddress(addr) {
		addr = ""
	}
	if _, err := s.Pool.Exec(ctx, `UPDATE nodes SET facts = $2, cached_artifacts = $3, agent_version = $4, last_seen_at = now(), updated_at = now(),
		address = CASE WHEN $5 <> '' THEN $5 ELSE address END WHERE id = $1`, nodeID, facts, cached, req.Facts.AgentVersion, addr); err != nil {
		return err
	}
	for _, rs := range req.Replicas {
		if !reportable[rs.State] {
			continue
		}
		msg := rs.Error
		if len(msg) > 2000 {
			msg = msg[:2000]
		}
		slots := min(max(rs.MaxBatch, 0), 128)
		// updated_at moves only when the state does: the scheduler's failure timers count from it
		if _, err := s.Pool.Exec(ctx, `UPDATE replicas SET state = $4, error = $5,
			updated_at = CASE WHEN state IS DISTINCT FROM $4 THEN now() ELSE updated_at END,
			slots = CASE WHEN $6 > 0 THEN $6 ELSE slots END,
			ready_at = CASE WHEN $4 = 'ready' AND state <> 'ready' THEN now() WHEN $4 <> 'ready' THEN NULL ELSE ready_at END
			WHERE id = $1 AND node_id = $2 AND generation = $3`, rs.ReplicaID, nodeID, rs.Generation, rs.State, msg, slots); err != nil {
			return err
		}
		if _, err := s.Pool.Exec(ctx, `UPDATE replica_members m SET state = $4, error = $5,
			updated_at = CASE WHEN m.state IS DISTINCT FROM $4 THEN now() ELSE m.updated_at END FROM replicas r
			WHERE m.replica_id = $1 AND m.node_id = $2 AND r.id = m.replica_id AND r.generation = $3`, rs.ReplicaID, nodeID, rs.Generation, rs.State, msg); err != nil {
			return err
		}
	}
	return nil
}

// desired is the node's full desired state; anything not listed must stop.
func (s *Server) desired(ctx context.Context, nodeID string) (wire.SyncResponse, error) {
	var resp wire.SyncResponse
	var offline bool
	if err := s.Pool.QueryRow(ctx, "SELECT n.status, n.sync_generation, s.offline FROM nodes n, settings s WHERE n.id = $1", nodeID).
		Scan(&resp.NodeStatus, &resp.Generation, &offline); err != nil {
		return resp, err
	}
	resp.Offline, resp.PermitKeys, resp.Assignments = offline, s.Trust.PublicKeys(), []wire.Assignment{}
	if resp.NodeStatus != "active" && resp.NodeStatus != "draining" {
		return resp, nil
	}
	rows, err := s.Pool.Query(ctx, `SELECT r.id, r.deployment_id, r.deployment_revision, r.generation, m.name, r.desired, r.slots, r.spec,
		coalesce(rm.devices, r.devices), coalesce(rm.member, 0), coalesce(r.rank_port, 0), coalesce(ln.facts->>'rank_host', ''), ln.address
		FROM replicas r JOIN deployments d ON d.id = r.deployment_id JOIN models m ON m.id = d.model_id JOIN nodes ln ON ln.id = r.node_id
		LEFT JOIN replica_members rm ON rm.replica_id = r.id AND rm.node_id = $1
		WHERE (r.node_id = $1 AND NOT (r.desired = 'stopped' AND r.state = 'stopped'))
			OR (rm.node_id = $1 AND NOT (r.desired = 'stopped' AND rm.state = 'stopped')) ORDER BY r.id`, nodeID)
	if err != nil {
		return resp, err
	}
	defer rows.Close()
	for rows.Next() {
		var a wire.Assignment
		var spec []byte
		var member, port int
		var rankHost, leadAddress string
		if err := rows.Scan(&a.ReplicaID, &a.DeploymentID, &a.Revision, &a.Generation, &a.ServedName, &a.Desired, &a.Slots, &spec, &a.Devices,
			&member, &port, &rankHost, &leadAddress); err != nil {
			return resp, err
		}
		if err := json.Unmarshal(spec, &a.Spec); err != nil {
			return resp, err
		}
		if a.Spec.Span() > 1 && port > 0 {
			if rankHost == "" {
				rankHost, _, _ = net.SplitHostPort(leadAddress)
			}
			a.Group = &wire.Group{Member: member, World: a.Spec.Ranks(), FirstRank: member * a.Spec.Width(), Master: rankHost, Port: port}
		}
		resp.Assignments = append(resp.Assignments, a)
	}
	return resp, rows.Err()
}
