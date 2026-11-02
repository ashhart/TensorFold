package admission

import (
	"context"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

type localCand struct {
	t     Target
	slots int
	used  int
}

type cloudCand struct {
	route catalog.ProviderRoute
	used  int
}

// candidates lists every eligible target with its current use; all holds every eligible id for queue overlap.
func candidates(ctx context.Context, tx db.Querier, req Request) ([]localCand, []cloudCand, []string, error) {
	var local []localCand
	var all []string
	if req.Local {
		rows, err := tx.Query(ctx, `SELECT r.id, r.node_id, n.address, d.id, d.revision, m.name, r.slots,
			(SELECT count(*) FROM leases l WHERE l.replica_id = r.id AND l.heartbeat_at > now() - interval '`+staleSQL+`')
			FROM replicas r JOIN deployments d ON d.id = r.deployment_id JOIN nodes n ON n.id = r.node_id JOIN models m ON m.id = d.model_id
			WHERE d.model_id = $1 AND d.status = 'active' AND r.desired = 'running' AND r.state = 'ready' AND r.deployment_revision = d.revision
			AND n.status = 'active' AND n.last_seen_at > now() - interval '90 seconds'
			AND array_position(ARRAY['public','internal','confidential','restricted'], n.max_classification) - 1 >= $2
			AND starlight_members_eligible(r.id, $2) AND (NOT $3 OR `+privateCache("$4")+`)
			ORDER BY r.id`, req.ModelID, int(req.Class), req.Isolated, nullID(req.Who.ProjectID))
		if err != nil {
			return nil, nil, nil, err
		}
		for rows.Next() {
			var c localCand
			if err := rows.Scan(&c.t.ReplicaID, &c.t.NodeID, &c.t.NodeAddress, &c.t.DeploymentID, &c.t.Revision, &c.t.ServedName, &c.slots, &c.used); err != nil {
				rows.Close()
				return nil, nil, nil, err
			}
			if pinned(req, c.t) && !excluded(req, c.t.ReplicaID) {
				local = append(local, c)
				all = append(all, c.t.ReplicaID)
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, nil, nil, err
		}
	}
	var cloud []cloudCand
	if len(req.Routes) > 0 {
		ids := make([]string, len(req.Routes))
		for i, r := range req.Routes {
			ids[i] = r.ID
		}
		used := map[string]int{}
		rows, err := tx.Query(ctx, `SELECT route_id, count(*) FROM leases WHERE route_id = ANY($1::uuid[])
			AND heartbeat_at > now() - interval '`+staleSQL+`' GROUP BY route_id`, ids)
		if err != nil {
			return nil, nil, nil, err
		}
		for rows.Next() {
			var id string
			var n int
			if err := rows.Scan(&id, &n); err != nil {
				rows.Close()
				return nil, nil, nil, err
			}
			used[id] = n
		}
		rows.Close()
		for _, r := range req.Routes {
			if excluded(req, r.ID) || req.Only != "" && req.Only != r.ID {
				continue
			}
			cloud = append(cloud, cloudCand{route: r, used: used[r.ID]})
			all = append(all, r.ID)
		}
	}
	return local, cloud, all, nil
}

func pinned(req Request, t Target) bool {
	if req.Only != "" && t.ReplicaID != req.Only {
		return false
	}
	for _, x := range req.Exclude {
		if x == t.ReplicaID {
			return false
		}
	}
	if len(req.Pins) == 0 {
		return true
	}
	for _, p := range req.Pins {
		if p.Deployment == t.DeploymentID && p.Revision == t.Revision {
			return true
		}
	}
	return false
}

// room reports whether a free slot or route is left once queued requests that outrank this one take theirs.
func (a *Admitter) room(ctx context.Context, q db.Querier, req Request, e *entry, local []localCand, cloud []cloudCand) (bool, error) {
	var freeIDs []string
	free := 0
	for _, c := range local {
		if n := c.slots - c.used; n > 0 {
			freeIDs, free = append(freeIDs, c.t.ReplicaID), free+n
		}
	}
	for _, c := range cloud {
		if n := c.route.MaxConcurrency - c.used; n > 0 {
			freeIDs, free = append(freeIDs, c.route.ID), free+n
		}
	}
	if free == 0 {
		return false, nil
	}
	blocked, err := a.outranked(ctx, q, req, e, freeIDs, free)
	return blocked < free, err
}

// outranked counts, up to limit, live queue entries that would beat this request to the free targets.
func (a *Admitter) outranked(ctx context.Context, q db.Querier, req Request, e *entry, freeIDs []string, limit int) (int, error) {
	id, at := noQueueEntry, (*time.Time)(nil)
	if e != nil {
		id, at = e.id, &e.enqueued
	}
	var n int
	err := q.QueryRow(ctx, outrankedSQL, req.ModelID, id, freeIDs, a.ageing().Seconds(), Rank(req.Priority), at, limit).Scan(&n)
	return n, err
}

// outrankedSQL counts older entries ranked at least as high and newer ones only from a higher class, all ranked in SQL.
const outrankedSQL = `SELECT count(*) FROM (
	(SELECT 1 FROM admission_queue q WHERE q.model_id = $1 AND q.id <> $2 AND q.enqueued_at < coalesce($6, now()) AND q.candidates && $3::uuid[]
		AND q.heartbeat_at > now() - interval '` + queueSQL + `' AND q.deadline > now()
		AND starlight_rank(q.base_rank, q.enqueued_at, $4) >= starlight_rank($5, coalesce($6, now()), $4) ORDER BY q.enqueued_at LIMIT $7)
	UNION ALL
	(SELECT 1 FROM admission_queue q WHERE q.model_id = $1 AND q.id <> $2 AND q.enqueued_at >= coalesce($6, now()) AND q.base_rank > $5
		AND q.candidates && $3::uuid[] AND q.heartbeat_at > now() - interval '` + queueSQL + `' AND q.deadline > now()
		AND starlight_rank(q.base_rank, q.enqueued_at, $4) > starlight_rank($5, coalesce($6, now()), $4) LIMIT $7)) s`

func (a *Admitter) placeable(ctx context.Context, req Request) (bool, error) {
	if len(req.Routes) > 0 {
		return true, nil
	}
	if !req.Local {
		return false, nil
	}
	ctx, cancel := context.WithTimeout(ctx, a.timeout())
	defer cancel()
	var ok bool
	if len(req.Pins) > 0 {
		for _, p := range req.Pins {
			err := a.Pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM deployments d WHERE d.id = $1 AND d.model_id = $2 AND d.revision = $3 "+
				"AND d.status = 'active' AND (NOT $4 OR "+privateCache("$5")+"))", p.Deployment, req.ModelID, p.Revision, req.Isolated,
				nullID(req.Who.ProjectID)).Scan(&ok)
			if err != nil || ok {
				return ok, err
			}
		}
		return false, nil
	}
	err := a.Pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM deployments d WHERE d.model_id = $1 AND d.status = 'active' AND (NOT $2 OR "+
		privateCache("$3")+"))", req.ModelID, req.Isolated, nullID(req.Who.ProjectID)).Scan(&ok)
	return ok, err
}

// privateCache holds where deployment d shares no prompt state: its cache is off, or only the project uses its model.
func privateCache(project string) string {
	return `(d.spec->>'prompt_cache' = 'off' OR NOT EXISTS (SELECT 1 FROM model_grants g WHERE g.model_id = d.model_id
		AND (g.project_id IS NULL OR g.project_id IS DISTINCT FROM ` + project + `::uuid)))`
}

func excluded(req Request, id string) bool {
	for _, x := range req.Exclude {
		if x == id {
			return true
		}
	}
	return false
}
