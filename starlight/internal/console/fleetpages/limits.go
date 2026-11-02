package fleetpages

import (
	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

type limitsPage struct {
	Limits []fleet.Limit
	Kinds  []string
	Models []fleet.Model
}

var orgScope = []string{"org"}

func (p *pages) limits() {
	p.c.Handle("GET /console/limits", authz.LimitsWrite, p.limitList)
	p.c.Handle("POST /console/limits", authz.LimitsWrite, p.limitSet)
	p.c.Handle("POST /console/limits/{id}/remove", authz.LimitsWrite, p.limitRemove)
}

func (p *pages) limitList(q *console.Request) error {
	list, err := fleet.Limits(q.Ctx(), p.c.Pool, orgScope, "")
	if err != nil {
		return err
	}
	models, err := fleet.Models(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	q.Render("fleet/limits", "Limits", limitsPage{Limits: list, Kinds: fleet.LimitKinds, Models: models})
	return nil
}

func (p *pages) limitSet(q *console.Request) error {
	in, err := limitInput(q, "org", "")
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetLimit(q.Ctx(), in) }); err != nil {
		return err
	}
	return q.Redirect("/console/limits", "Limit saved; it applies to new requests at once.")
}

func (p *pages) limitRemove(q *console.Request) error {
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.RemoveLimit(q.Ctx(), q.R.PathValue("id"), orgScope, "") }); err != nil {
		return err
	}
	return q.Redirect("/console/limits", "Limit removed.")
}
