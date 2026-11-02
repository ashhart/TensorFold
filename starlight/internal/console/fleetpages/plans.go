package fleetpages

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

type plansPage struct {
	Pending    []fleet.Plan
	Decided    []fleet.Plan
	Mine       []fleet.Plan
	CanApprove bool
}

type planPage struct {
	format
	Plan       fleet.Plan
	Text       string
	TextSHA256 string
	Intact     bool
	Owner      bool
	CanDecide  bool
	ReadOnly   bool
}

var decided = []string{"approved", "rejected", "failed", "revoked"}

func (p *pages) planPages() {
	p.c.Handle("GET /console/plans", "", p.planList)
	p.c.Handle("GET /console/plans/{id}", "", p.planDetail)
	p.c.Handle("POST /console/plans/{id}/approve", "", p.planApprove)
	p.c.Handle("POST /console/plans/{id}/reject", "", p.planReject)
}

func approverProjects(q *console.Request) []string {
	return visibleProjects(q, authz.ProjectApprove, authz.ProjectApprove)
}

func (p *pages) planList(q *console.Request) error {
	ctx, pool := q.Ctx(), p.c.Pool
	only := approverProjects(q)
	page := plansPage{CanApprove: only == nil || len(only) > 0}
	var err error
	if page.CanApprove {
		if page.Pending, err = fleet.Plans(ctx, pool, fleet.PlanFilter{Projects: only, Statuses: []string{"pending_approval"}}); err != nil {
			return err
		}
		if page.Decided, err = fleet.Plans(ctx, pool, fleet.PlanFilter{Projects: only, Statuses: decided, Limit: 25}); err != nil {
			return err
		}
	}
	if page.Mine, err = fleet.Plans(ctx, pool, fleet.PlanFilter{OwnerID: q.User.ID, Limit: 50}); err != nil {
		return err
	}
	q.Render("fleet/plans", "Plans", page)
	return nil
}

// plan loads a plan the user owns or may approve; anyone else gets not found, so plan ids reveal nothing.
func (p *pages) plan(q *console.Request) (fleet.Plan, bool, error) {
	pl, err := fleet.PlanByID(q.Ctx(), p.c.Pool, q.R.PathValue("id"))
	if err != nil {
		return pl, false, friendly(err)
	}
	owner := pl.OwnerType == "user" && pl.OwnerID == q.User.ID
	if !owner && !q.Can(authz.ProjectApprove, pl.ProjectID) {
		return pl, false, friendly(fleet.ErrNotFound)
	}
	return pl, owner, nil
}

func (p *pages) planDetail(q *console.Request) error {
	pl, owner, err := p.plan(q)
	if err != nil {
		return err
	}
	page := planPage{Plan: pl, Owner: owner, ReadOnly: p.plans == nil}
	if len(pl.Sealed) > 0 {
		text, err := p.c.Keys.Open(pl.Sealed, "plans.plan:"+pl.ID)
		if err != nil {
			return err
		}
		sum := sha256.Sum256(text)
		page.Text, page.TextSHA256 = string(text), hex.EncodeToString(sum[:])
		page.Intact = page.TextSHA256 == pl.PlanSHA256
	}
	page.CanDecide = page.Intact && pl.Status == "pending_approval" && p.plans != nil && q.Can(authz.ProjectApprove, pl.ProjectID)
	q.Render("fleet/plan", pl.Title, page)
	return nil
}

func (p *pages) decide(q *console.Request, f func(fleet.Plan) error, notice string) error {
	pl, _, err := p.plan(q)
	if err != nil {
		return err
	}
	if !q.Can(authz.ProjectApprove, pl.ProjectID) {
		return forbidden()
	}
	if p.plans == nil {
		return httpx.Errorf(http.StatusNotImplemented, "not_enabled", "Deciding plans from the console is not enabled on this server.")
	}
	if err := f(pl); err != nil {
		return err
	}
	return q.Redirect("/console/plans/"+pl.ID, notice)
}

func (p *pages) planApprove(q *console.Request) error {
	return p.decide(q, func(pl fleet.Plan) error {
		return p.plans.Approve(q.R, caller(q), pl.ID, q.Form("plan_sha256"))
	}, "Plan approved. The owner can now start the local task.")
}

func (p *pages) planReject(q *console.Request) error {
	return p.decide(q, func(pl fleet.Plan) error { return p.plans.Reject(q.R, caller(q), pl.ID) }, "Plan rejected.")
}
