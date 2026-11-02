package gateway

import (
	"context"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

var responseID = regexp.MustCompile(`^[A-Za-z0-9_-]{1,128}$`)

type owner struct {
	principalType, principalID, projectID string
	replica, route                        *string
}

func (g *Gateway) owner(ctx context.Context, id string) (owner, bool) {
	var o owner
	err := g.Pool.QueryRow(ctx, `SELECT principal_type, principal_id, project_id, replica_id, route_id FROM response_objects
		WHERE response_id = $1 AND expires_at > now()`, id).Scan(&o.principalType, &o.principalID, &o.projectID, &o.replica, &o.route)
	return o, err == nil
}

func (c *call) owns(o owner) bool {
	p := c.caller.P
	return o.principalType == string(p.Type) && o.principalID == p.ID && o.projectID == p.ProjectID
}

// continuation pins a previous_response_id request to the replica or route that stored it, for its owner only.
func (g *Gateway) continuation(ctx context.Context, c *call) error {
	if c.protocol != providers.Responses || c.info.PreviousID == "" {
		return nil
	}
	o, ok := g.owner(ctx, c.info.PreviousID)
	if !ok || !c.owns(o) {
		e := httpx.Errorf(http.StatusNotFound, "previous_response_not_found", "previous_response_id was not found")
		e.Param = "previous_response_id"
		return e
	}
	if o.replica != nil {
		c.req.Only = *o.replica
		c.plan.cloud = nil
	} else {
		c.req.Only = *o.route
		c.plan.local = false
	}
	return nil
}

// remember records who owns a stored response so only they can read, delete or continue it.
func (g *Gateway) remember(ctx context.Context, c *call, l *admission.Lease, out relayed) {
	if c.protocol != providers.Responses || !c.info.Store || out.status != http.StatusOK || !responseID.MatchString(out.responseID) {
		return
	}
	var replica, route *string
	if l.Target.Local() {
		replica = &l.Target.ReplicaID
	} else {
		route = &l.Target.Route.ID
	}
	p := c.caller.P
	_, _ = g.Pool.Exec(context.WithoutCancel(ctx), `INSERT INTO response_objects (response_id, principal_type, principal_id, project_id, replica_id,
		route_id, expires_at) VALUES ($1,$2,$3,$4,$5,$6, now() + interval '7 days') ON CONFLICT (response_id) DO NOTHING`,
		out.responseID, string(p.Type), p.ID, p.ProjectID, replica, route)
}

// storedResponse serves GET and DELETE /v1/responses/{id} for the response's owner, from where it is stored.
func (g *Gateway) storedResponse(w http.ResponseWriter, r *http.Request) {
	caller, _, _, _, err := g.authenticate(r)
	if err != nil {
		g.refuse(w, httpx.OpenAI, err)
		return
	}
	id := r.PathValue("id")
	c := &call{caller: caller}
	o, ok := g.owner(r.Context(), id)
	if !responseID.MatchString(id) || !ok || !c.owns(o) {
		httpx.Write(w, httpx.OpenAI, httpx.Errorf(http.StatusNotFound, "not_found", "No response found with id '"+id+"'."))
		return
	}
	var resp *http.Response
	if o.replica != nil {
		resp, err = g.fromReplica(r.Context(), *o.replica, r.Method, "/v1/responses/"+id)
	} else {
		resp, err = g.fromRoute(r.Context(), *o.route, r.Method, "/responses/"+id)
	}
	if err != nil {
		httpx.Write(w, httpx.OpenAI, httpx.Errorf(http.StatusBadGateway, "upstream_unreachable", "the server holding this response did not answer"))
		return
	}
	defer resp.Body.Close()
	if r.Method == http.MethodDelete && resp.StatusCode < 300 {
		_, _ = g.Pool.Exec(r.Context(), "DELETE FROM response_objects WHERE response_id = $1", id)
	}
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(resp.Body, maxResponse))
}

func (g *Gateway) fromReplica(ctx context.Context, replicaID, method, path string) (*http.Response, error) {
	var t admission.Target
	err := g.Pool.QueryRow(ctx, `SELECT r.id, r.node_id, n.address, d.id, d.revision, m.name FROM replicas r
		JOIN nodes n ON n.id = r.node_id JOIN deployments d ON d.id = r.deployment_id JOIN models m ON m.id = d.model_id
		WHERE r.id = $1 AND n.status = 'active'`, replicaID).Scan(&t.ReplicaID, &t.NodeID, &t.NodeAddress, &t.DeploymentID, &t.Revision, &t.ServedName)
	if db.NotFound(err) {
		return &http.Response{StatusCode: http.StatusNotFound, Header: http.Header{"Content-Type": {"application/json"}},
			Body: io.NopCloser(strings.NewReader(`{"error":{"message":"the replica holding this response is gone","type":"not_found_error","code":"not_found"}}`))}, nil
	}
	if err != nil {
		return nil, err
	}
	d, err := g.Nodes.Do(ctx, &admission.Lease{ID: ids.New(), Target: t}, method, path, nil, "", "standard", 0)
	return d.Resp, err
}

func (g *Gateway) fromRoute(ctx context.Context, routeID, method, path string) (*http.Response, error) {
	route, err := g.Catalog.Route(ctx, routeID)
	if err != nil {
		return nil, err
	}
	return g.Providers.Fetch(ctx, route, method, path)
}
