package fleetpages

import (
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/control"
)

type nodesPage struct {
	format
	Nodes      []fleet.NodeLive
	Tokens     []fleet.EnrolmentToken
	CanControl bool
	Next       string
	NodeURL    string
}

type nodePage struct {
	format
	Node       fleet.Node
	Live       fleet.NodeLive
	Labels     string
	Replicas   []fleet.Replica
	CanControl bool
	Next       string
	Classes    []string
}

var classes = []string{"public", "internal", "confidential", "restricted"}

func (p *pages) nodes() {
	p.c.Handle("GET /console/nodes", authz.NodesRead, p.nodeList)
	p.c.Handle("POST /console/nodes/tokens", authz.NodesWrite, p.createToken)
	p.c.Handle("POST /console/nodes/tokens/{id}/revoke", authz.NodesWrite, p.revokeToken)
	p.c.Handle("GET /console/nodes/{name}", authz.NodesRead, p.nodeDetail)
	p.c.Handle("POST /console/nodes/{name}", authz.NodesWrite, p.nodeEdit)
	p.c.Handle("POST /console/nodes/{name}/action", authz.NodesWrite, p.nodeAction)
}

func (p *pages) nodeURL() string {
	if p.c.NodeURL != "" {
		return p.c.NodeURL
	}
	return "https://<starlight-host>:9443"
}

func (p *pages) nodeList(q *console.Request) error {
	nodes, err := fleet.Nodes(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	tokens, err := fleet.EnrolmentTokens(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	page := nodesPage{Tokens: tokens, CanControl: q.Can(authz.NodesWrite, ""), Next: "/console/nodes", NodeURL: p.nodeURL()}
	for _, n := range nodes {
		page.Nodes = append(page.Nodes, n.Row())
	}
	q.Render("fleet/nodes", "Nodes", page)
	return nil
}

func (p *pages) createToken(q *console.Request) error {
	lbl, err := parseLabels(q.Lines("labels"))
	if err != nil {
		return err
	}
	uses, err := q.Int("uses", 1)
	if err != nil {
		return err
	}
	hours, err := q.Int("ttl_hours", 24)
	if err != nil {
		return err
	}
	spec := control.TokenSpec{Pool: q.Form("pool"), Labels: lbl, MaxClass: q.Form("max_class"), AutoApprove: q.Bool("auto_approve"), MaxUses: uses,
		TTL: time.Duration(hours) * time.Hour}
	var tok string
	if err := p.do(q, func(tx *fleet.Tx) (err error) { tok, err = tx.CreateEnrolmentToken(q.Ctx(), spec); return err }); err != nil {
		return err
	}
	cmd := "starlight-node enroll --server " + p.nodeURL() + " --token " + tok + " --ca-sha256 " + p.c.Trust.Fingerprint()
	q.Render("fleet/secret", "Enrolment token", secret{Label: "Run this on the machine you are enrolling. The token is shown only once.",
		Value: tok, Command: cmd, Back: "/console/nodes"})
	return nil
}

func (p *pages) revokeToken(q *console.Request) error {
	id := q.R.PathValue("id")
	if !validID(id) {
		return friendly(fleet.ErrNotFound)
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.RevokeEnrolmentToken(q.Ctx(), id) }); err != nil {
		return err
	}
	return q.Redirect("/console/nodes", "Enrolment token revoked.")
}

func (p *pages) nodeDetail(q *console.Request) error {
	n, err := fleet.NodeByName(q.Ctx(), p.c.Pool, q.R.PathValue("name"))
	if err != nil {
		return friendly(err)
	}
	reps, err := fleet.Replicas(q.Ctx(), p.c.Pool, "", n.ID)
	if err != nil {
		return err
	}
	q.Render("fleet/node", n.Name, nodePage{Node: n, Live: n.Row(), Labels: joinPairs(n.Labels, "="), Replicas: reps,
		CanControl: q.Can(authz.NodesWrite, ""), Next: "/console/nodes/" + n.Name, Classes: classes})
	return nil
}

func (p *pages) nodeEdit(q *console.Request) error {
	name := q.R.PathValue("name")
	lbl, err := parseLabels(q.Lines("labels"))
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error {
		return tx.UpdateNode(q.Ctx(), name, q.Form("pool"), lbl, q.Form("max_class"))
	}); err != nil {
		return err
	}
	return q.Redirect("/console/nodes/"+name, "Node updated.")
}

var actionNotices = map[string]string{fleet.NodeApprove: "Node approved.", fleet.NodeOnline: "Node is back online.",
	fleet.NodeDrain: "Node is draining; its replicas move elsewhere.", fleet.NodeDisable: "Node taken offline.",
	fleet.NodeRevoke: "Node revoked and its certificates withdrawn."}

func (p *pages) nodeAction(q *console.Request) error {
	name, action := q.R.PathValue("name"), q.Form("action")
	if action == fleet.NodeRevoke && !q.Bool("confirm") {
		return console.Invalid("Tick the confirmation box to revoke a node; revoking cannot be undone.")
	}
	if _, ok := actionNotices[action]; !ok {
		return console.Invalid("Unknown node action.")
	}
	if err := p.do(q, func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(q.Ctx(), name, action); return err }); err != nil {
		return err
	}
	next := q.Form("next")
	if next != "/console" && next != "/console/nodes" && next != "/console/nodes/"+name {
		next = "/console/nodes"
	}
	return q.Redirect(next, actionNotices[action])
}
