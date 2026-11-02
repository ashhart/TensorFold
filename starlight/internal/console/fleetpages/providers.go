package fleetpages

import (
	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

type providerRoute struct {
	ModelName string
	fleet.Route
}

type providersPage struct {
	Providers []fleet.Provider
	Kinds     []fleet.ProviderKind
	Classes   []string
	CanWrite  bool
}

type providerPage struct {
	Provider fleet.Provider
	Headers  string
	Routes   []providerRoute
	Kinds    []fleet.ProviderKind
	Classes  []string
	CanWrite bool
}

func (p *pages) providers() {
	p.c.Handle("GET /console/providers", authz.ProvidersRead, p.providerList)
	p.c.Handle("POST /console/providers", authz.ProvidersWrite, p.providerCreate)
	p.c.Handle("GET /console/providers/{name}", authz.ProvidersRead, p.providerDetail)
	p.c.Handle("POST /console/providers/{name}", authz.ProvidersWrite, p.providerUpdate)
}

func providerInput(q *console.Request, name string) (fleet.ProviderInput, error) {
	headers, err := fleet.ParseHeaders(q.Lines("headers"))
	if err != nil {
		return fleet.ProviderInput{}, friendly(err)
	}
	return fleet.ProviderInput{Name: name, Kind: q.Form("kind"), BaseURL: q.Form("base_url"), Credential: q.Form("credential"),
		ClearCredential: q.Bool("clear_credential"), MaxClass: q.Form("max_class"), Region: q.Form("region"), Headers: headers,
		Internal: q.Bool("internal"), Enabled: q.Bool("enabled")}, nil
}

func (p *pages) providerList(q *console.Request) error {
	list, err := fleet.Providers(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	q.Render("fleet/providers", "Providers", providersPage{Providers: list, Kinds: fleet.ProviderKinds, Classes: classes,
		CanWrite: q.Can(authz.ProvidersWrite, "")})
	return nil
}

func (p *pages) providerCreate(q *console.Request) error {
	in, err := providerInput(q, q.Form("name"))
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { _, err := tx.CreateProvider(q.Ctx(), in); return err }); err != nil {
		return err
	}
	return q.Redirect("/console/providers/"+in.Name, "Provider added. Route models to it from each model's page.")
}

func (p *pages) providerDetail(q *console.Request) error {
	ctx, pool := q.Ctx(), p.c.Pool
	pr, err := fleet.ProviderByName(ctx, pool, q.R.PathValue("name"))
	if err != nil {
		return friendly(err)
	}
	routes, err := fleet.Routes(ctx, pool, "")
	if err != nil {
		return err
	}
	models, err := fleet.Models(ctx, pool)
	if err != nil {
		return err
	}
	names := map[string]string{}
	for _, m := range models {
		names[m.ID] = m.Name
	}
	page := providerPage{Provider: pr, Headers: joinPairs(pr.Headers, ": "), Kinds: fleet.ProviderKinds, Classes: classes,
		CanWrite: q.Can(authz.ProvidersWrite, "")}
	for _, r := range routes {
		if r.ProviderID == pr.ID {
			page.Routes = append(page.Routes, providerRoute{ModelName: names[r.ModelID], Route: r})
		}
	}
	q.Render("fleet/provider", pr.Name, page)
	return nil
}

func (p *pages) providerUpdate(q *console.Request) error {
	pr, err := fleet.ProviderByName(q.Ctx(), p.c.Pool, q.R.PathValue("name"))
	if err != nil {
		return friendly(err)
	}
	in, err := providerInput(q, pr.Name)
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { _, err := tx.UpdateProvider(q.Ctx(), pr.ID, in); return err }); err != nil {
		return err
	}
	return q.Redirect("/console/providers/"+pr.Name, "Provider saved.")
}
