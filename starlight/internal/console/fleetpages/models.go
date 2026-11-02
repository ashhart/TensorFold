package fleetpages

import (
	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

type modelsPage struct {
	format
	Models   []fleet.Model
	Live     map[string]fleet.ModelLive
	CanWrite bool
}

type modelPage struct {
	format
	Model       fleet.Model
	Live        fleet.ModelLive
	Aliases     []fleet.Alias
	OrgGrant    *fleet.Grant
	Grants      []fleet.Grant
	Routes      []fleet.Route
	Providers   []fleet.Provider
	Protocols   []string
	Policies    []string
	Deployments []fleet.Deployment
	CanWrite    bool
	CanRoutes   bool
}

var routePolicies = []string{"local_only", "local_first", "cloud_first", "cloud_only"}

var allProtocols = []string{providers.ChatCompletions, providers.Completions, providers.Embeddings, providers.Responses, providers.Messages,
	providers.CountTokens}

func (p *pages) models() {
	p.c.Handle("GET /console/models", authz.ModelsRead, p.modelList)
	p.c.Handle("POST /console/models", authz.ModelsWrite, p.modelCreate)
	p.c.Handle("GET /console/models/{id}", authz.ModelsRead, p.modelDetail)
	for path, h := range map[string]func(*console.Request, fleet.Model) error{
		"": p.modelUpdate, "/aliases": p.globalAlias, "/aliases/remove": p.globalAliasRemove, "/grant": p.orgGrant,
		"/routes": p.routeSet, "/routes/{route}/remove": p.routeRemove, "/deployments": p.deploymentCreate,
	} {
		h := h
		p.c.Handle("POST /console/models/{id}"+path, authz.ModelsWrite, func(q *console.Request) error {
			m, err := fleet.ModelByID(q.Ctx(), p.c.Pool, q.R.PathValue("id"))
			if err != nil {
				return friendly(err)
			}
			return h(q, m)
		})
	}
}

func (p *pages) modelList(q *console.Request) error {
	list, err := fleet.Models(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	live, err := fleet.ModelsLive(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	byID := map[string]fleet.ModelLive{}
	for _, m := range live {
		byID[m.ID] = m
	}
	q.Render("fleet/models", "Models", modelsPage{Models: list, Live: byID, CanWrite: q.Can(authz.ModelsWrite, "")})
	return nil
}

func modelInput(q *console.Request, name string) (fleet.ModelInput, error) {
	ctxWin, err := q.Int("context_window", 0)
	if err != nil {
		return fleet.ModelInput{}, err
	}
	in := fleet.ModelInput{Name: name, DisplayName: q.Form("display_name"), Description: q.Form("description"), Kind: q.Form("kind"),
		RoutePolicy: q.Form("route_policy"), Status: q.Form("status"),
		Capabilities: fleet.Capabilities{ContextWindow: ctxWin, Tools: q.Bool("tools"), Vision: q.Bool("vision"), Reasoning: q.Bool("reasoning")}}
	in.Chargeback, err = prices(q)
	return in, err
}

// prices reads dollars-per-million-token fields into micro-dollars.
func prices(q *console.Request) (fleet.Prices, error) {
	var pr fleet.Prices
	var err error
	if pr.Input, err = fleet.ParseDollars("input price", q.Form("price_input")); err != nil {
		return pr, friendly(err)
	}
	if pr.Output, err = fleet.ParseDollars("output price", q.Form("price_output")); err != nil {
		return pr, friendly(err)
	}
	if pr.Cached, err = fleet.ParseDollars("cached input price", q.Form("price_cached")); err != nil {
		return pr, friendly(err)
	}
	return pr, nil
}

func (p *pages) modelCreate(q *console.Request) error {
	in, err := modelInput(q, q.Form("name"))
	if err != nil {
		return err
	}
	var m fleet.Model
	if err := p.do(q, func(tx *fleet.Tx) (err error) { m, err = tx.CreateModel(q.Ctx(), in); return err }); err != nil {
		return err
	}
	return q.Redirect("/console/models/"+m.ID, "Model added to the catalogue.")
}

func (p *pages) modelDetail(q *console.Request) error {
	ctx, pool := q.Ctx(), p.c.Pool
	m, err := fleet.ModelByID(ctx, pool, q.R.PathValue("id"))
	if err != nil {
		return friendly(err)
	}
	page := modelPage{Model: m, Protocols: allProtocols, Policies: routePolicies, CanWrite: q.Can(authz.ModelsWrite, ""), CanRoutes: q.Can(authz.ProvidersRead, "")}
	live, err := fleet.ModelsLive(ctx, pool)
	if err != nil {
		return err
	}
	for _, l := range live {
		if l.ID == m.ID {
			page.Live = l
		}
	}
	all, err := fleet.Aliases(ctx, pool, "global")
	if err != nil {
		return err
	}
	for _, a := range all {
		if a.ModelID == m.ID {
			page.Aliases = append(page.Aliases, a)
		}
	}
	grants, err := fleet.Grants(ctx, pool, "", m.ID)
	if err != nil {
		return err
	}
	for i, g := range grants {
		if g.ProjectID == "" {
			page.OrgGrant = &grants[i]
		} else {
			page.Grants = append(page.Grants, g)
		}
	}
	if page.Routes, err = fleet.Routes(ctx, pool, m.ID); err != nil {
		return err
	}
	if page.CanWrite {
		if page.Providers, err = fleet.Providers(ctx, pool); err != nil {
			return err
		}
	}
	if page.Deployments, err = fleet.Deployments(ctx, pool, m.ID); err != nil {
		return err
	}
	q.Render("fleet/model", m.Name, page)
	return nil
}

func (p *pages) modelUpdate(q *console.Request, m fleet.Model) error {
	in, err := modelInput(q, m.Name)
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { _, err := tx.UpdateModel(q.Ctx(), m.ID, in); return err }); err != nil {
		return err
	}
	return q.Redirect("/console/models/"+m.ID, "Model saved.")
}

func (p *pages) globalAlias(q *console.Request, m fleet.Model) error {
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetAlias(q.Ctx(), q.Form("alias"), "", m.ID) }); err != nil {
		return err
	}
	return q.Redirect("/console/models/"+m.ID+"#aliases", "Alias saved.")
}

func (p *pages) globalAliasRemove(q *console.Request, m fleet.Model) error {
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.RemoveAlias(q.Ctx(), q.Form("alias"), "") }); err != nil {
		return err
	}
	return q.Redirect("/console/models/"+m.ID+"#aliases", "Alias removed.")
}

func (p *pages) orgGrant(q *console.Request, m fleet.Model) error {
	err := p.do(q, func(tx *fleet.Tx) error {
		if !q.Bool("everyone") {
			grants, err := fleet.Grants(q.Ctx(), tx.Q(), "", m.ID)
			if err != nil {
				return err
			}
			for _, g := range grants {
				if g.ProjectID == "" {
					return tx.RemoveGrant(q.Ctx(), g.ID)
				}
			}
			return nil
		}
		maxCtx, err := optInt(q, "max_context")
		if err != nil {
			return err
		}
		maxOut, err := optInt(q, "max_output_tokens")
		if err != nil {
			return err
		}
		return tx.SetGrant(q.Ctx(), m.ID, "", maxCtx, maxOut)
	})
	if err != nil {
		return err
	}
	return q.Redirect("/console/models/"+m.ID+"#access", "Access saved.")
}

func (p *pages) routeSet(q *console.Request, m fleet.Model) error {
	pr, err := prices(q)
	if err != nil {
		return err
	}
	priority, err := q.Int("priority", 100)
	if err != nil {
		return err
	}
	conc, err := q.Int("max_concurrency", 64)
	if err != nil {
		return err
	}
	var protocols []string
	for _, proto := range allProtocols {
		if q.Bool("protocol_" + proto) {
			protocols = append(protocols, proto)
		}
	}
	in := fleet.RouteInput{ProviderID: q.Form("provider"), UpstreamModel: q.Form("upstream_model"), Protocols: protocols, Priority: priority,
		Prices: pr, MaxConcurrency: conc, Enabled: q.Form("enabled") != "off"}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetRoute(q.Ctx(), m.ID, in) }); err != nil {
		return err
	}
	return q.Redirect("/console/models/"+m.ID+"#routes", "Route saved.")
}

func (p *pages) routeRemove(q *console.Request, m fleet.Model) error {
	routes, err := fleet.Routes(q.Ctx(), p.c.Pool, m.ID)
	if err != nil {
		return err
	}
	id := q.R.PathValue("route")
	for _, r := range routes {
		if r.ID == id {
			if err := p.do(q, func(tx *fleet.Tx) error { return tx.RemoveRoute(q.Ctx(), id) }); err != nil {
				return err
			}
			return q.Redirect("/console/models/"+m.ID+"#routes", "Route removed.")
		}
	}
	return friendly(fleet.ErrNotFound)
}
