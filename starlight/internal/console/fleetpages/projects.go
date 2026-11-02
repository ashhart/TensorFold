package fleetpages

import (
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

type projectsPage struct {
	Projects  []fleet.Project
	CanCreate bool
}

type projectPage struct {
	format
	Project     fleet.Project
	Members     []fleet.Binding
	Groups      []fleet.GroupChoice
	Grants      []fleet.Grant
	Aliases     []fleet.Alias
	Models      []fleet.Model
	Limits      []fleet.Limit
	Kinds       []string
	Accounts    []fleet.ServiceAccount
	Keys        []keys.Key
	Owners      map[string]string
	Usage       []fleet.UsageRow
	UsageTotal  fleet.UsageRow
	Classes     []string
	CanSettings bool
	CanMembers  bool
	CanModels   bool
	CanLimits   bool
	CanKeys     bool
	CanUsage    bool
	Now         time.Time
}

// IsCost reports whether a limit kind is money, for form hints.
func (projectPage) IsCost(kind string) bool { return fleet.IsCost(kind) }

func (p *pages) projects() {
	p.c.Handle("GET /console/projects", "", p.projectList)
	p.c.Handle("POST /console/projects", authz.ProjectsWrite, p.projectCreate)
	p.c.Handle("GET /console/projects/{slug}", "", p.projectDetail)
	for path, h := range map[string]func(*console.Request, fleet.Project) error{
		"settings": p.projectSettings, "archive": p.projectArchive, "members": p.memberAdd, "members/{id}/remove": p.memberRemove,
		"grants": p.grantSet, "grants/{id}/remove": p.grantRemove, "aliases": p.aliasSet, "aliases/remove": p.aliasRemove,
		"limits": p.projectLimitSet, "limits/{id}/remove": p.projectLimitRemove, "accounts": p.accountCreate,
		"accounts/{id}/status": p.accountStatus, "accounts/{id}/keys": p.accountKey, "keys/{id}/revoke": p.keyRevoke, "keys/{id}/rotate": p.keyRotate,
	} {
		h := h
		p.c.Handle("POST /console/projects/{slug}/"+path, "", func(q *console.Request) error {
			pr, err := p.project(q)
			if err != nil {
				return err
			}
			return h(q, pr)
		})
	}
}

// project loads the path's project; people who cannot view it get not-found so project names do not leak.
func (p *pages) project(q *console.Request) (fleet.Project, error) {
	pr, err := fleet.ProjectBySlug(q.Ctx(), p.c.Pool, q.R.PathValue("slug"))
	if err != nil || !q.Can(authz.ProjectView, pr.ID) && !q.Can(authz.ProjectsRead, "") {
		return pr, friendly(fleet.ErrNotFound)
	}
	return pr, nil
}

func need(q *console.Request, perm authz.Perm, project string) error {
	if !q.Can(perm, project) {
		return forbidden()
	}
	return nil
}

func (p *pages) projectList(q *console.Request) error {
	list, err := fleet.Projects(q.Ctx(), p.c.Pool, visibleProjects(q, authz.ProjectsRead, authz.ProjectView))
	if err != nil {
		return err
	}
	q.Render("fleet/projects", "Projects", projectsPage{Projects: list, CanCreate: q.Can(authz.ProjectsWrite, "")})
	return nil
}

func projectInput(q *console.Request, slug string) fleet.ProjectInput {
	return fleet.ProjectInput{Slug: slug, Name: q.Form("name"), Description: q.Form("description"), Classification: q.Form("classification"),
		MaxPriority: q.Form("max_priority"), Everyone: q.Bool("everyone"), PlanApproval: q.Form("plan_approval"), SecretsGuard: q.Form("secrets_guard"),
		IsolateCache: q.Bool("isolate_prompt_cache")}
}

func (p *pages) projectCreate(q *console.Request) error {
	var pr fleet.Project
	if err := p.do(q, func(tx *fleet.Tx) (err error) {
		pr, err = tx.CreateProject(q.Ctx(), projectInput(q, q.Form("slug")))
		return err
	}); err != nil {
		return err
	}
	return q.Redirect("/console/projects/"+pr.Slug, "Project created.")
}

func (p *pages) projectDetail(q *console.Request) error {
	pr, err := p.project(q)
	if err != nil {
		return err
	}
	ctx, pool := q.Ctx(), p.c.Pool
	page := projectPage{Project: pr, Kinds: fleet.LimitKinds, Classes: classes, Now: time.Now(), CanSettings: q.Can(authz.ProjectsWrite, ""),
		CanMembers: q.Can(authz.ProjectMembers, pr.ID), CanModels: q.Can(authz.ModelsWrite, ""), CanLimits: q.Can(authz.ProjectLimits, pr.ID),
		CanKeys: q.Can(authz.ProjectKeys, pr.ID), CanUsage: q.Can(authz.ProjectUsage, pr.ID)}
	if page.Members, err = fleet.Bindings(ctx, pool, pr.ID); err != nil {
		return err
	}
	if page.CanMembers {
		if page.Groups, err = fleet.GroupChoices(ctx, pool); err != nil {
			return err
		}
	}
	if page.Grants, err = fleet.Grants(ctx, pool, pr.ID, ""); err != nil {
		return err
	}
	if page.Aliases, err = fleet.Aliases(ctx, pool, pr.ID); err != nil {
		return err
	}
	if page.CanModels {
		if page.Models, err = fleet.Models(ctx, pool); err != nil {
			return err
		}
	}
	if page.Limits, err = fleet.Limits(ctx, pool, []string{"project", "member"}, pr.ID); err != nil {
		return err
	}
	if page.CanKeys {
		if page.Accounts, err = fleet.ServiceAccounts(ctx, pool, pr.ID); err != nil {
			return err
		}
		if page.Keys, page.Owners, err = fleet.ProjectKeys(ctx, pool, pr.ID); err != nil {
			return err
		}
	}
	if page.CanUsage {
		now := time.Now()
		f := fleet.UsageFilter{From: now.AddDate(0, 0, -30), To: now.Add(time.Minute), Projects: []string{pr.ID}, GroupBy: "model"}
		if page.Usage, page.UsageTotal, err = fleet.Usage(ctx, pool, f); err != nil {
			return err
		}
	}
	q.Render("fleet/project", pr.Name, page)
	return nil
}

func back(q *console.Request, pr fleet.Project, anchor, notice string) error {
	return q.Redirect("/console/projects/"+pr.Slug+anchor, notice)
}

func (p *pages) projectSettings(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectsWrite, ""); err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error {
		_, err := tx.UpdateProject(q.Ctx(), pr.ID, projectInput(q, pr.Slug))
		return err
	}); err != nil {
		return err
	}
	return back(q, pr, "", "Project settings saved.")
}

func (p *pages) projectArchive(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectsWrite, ""); err != nil {
		return err
	}
	archive := pr.ArchivedAt == nil
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetArchived(q.Ctx(), pr.ID, archive) }); err != nil {
		return err
	}
	if archive {
		return back(q, pr, "", "Project archived; it grants nothing until restored.")
	}
	return back(q, pr, "", "Project restored.")
}

func (p *pages) memberAdd(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectMembers, pr.ID); err != nil {
		return err
	}
	role := q.Form("role")
	err := p.do(q, func(tx *fleet.Tx) error {
		st, id := "group", q.Form("group")
		if email := q.Form("email"); email != "" {
			u, err := fleet.UserByEmail(q.Ctx(), tx.Q(), email)
			if err != nil {
				return err
			}
			st, id = "user", u.ID
		}
		_, err := tx.Bind(q.Ctx(), role, st, id, pr.ID)
		return err
	})
	if err != nil {
		return err
	}
	return back(q, pr, "#members", "Member added.")
}

func (p *pages) memberRemove(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectMembers, pr.ID); err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.Unbind(q.Ctx(), q.R.PathValue("id"), pr.ID) }); err != nil {
		return err
	}
	return back(q, pr, "#members", "Member removed.")
}

func (p *pages) grantSet(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ModelsWrite, ""); err != nil {
		return err
	}
	maxCtx, err := optInt(q, "max_context")
	if err != nil {
		return err
	}
	maxOut, err := optInt(q, "max_output_tokens")
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetGrant(q.Ctx(), q.Form("model"), pr.ID, maxCtx, maxOut) }); err != nil {
		return err
	}
	return back(q, pr, "#models", "Model granted.")
}

func (p *pages) grantRemove(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ModelsWrite, ""); err != nil {
		return err
	}
	gs, err := fleet.Grants(q.Ctx(), p.c.Pool, pr.ID, "")
	if err != nil {
		return err
	}
	id := q.R.PathValue("id")
	for _, g := range gs {
		if g.ID == id {
			if err := p.do(q, func(tx *fleet.Tx) error { return tx.RemoveGrant(q.Ctx(), id) }); err != nil {
				return err
			}
			return back(q, pr, "#models", "Grant removed.")
		}
	}
	return friendly(fleet.ErrNotFound)
}

func (p *pages) aliasSet(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ModelsWrite, ""); err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetAlias(q.Ctx(), q.Form("alias"), pr.ID, q.Form("model")) }); err != nil {
		return err
	}
	return back(q, pr, "#models", "Alias saved.")
}

func (p *pages) aliasRemove(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ModelsWrite, ""); err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.RemoveAlias(q.Ctx(), q.Form("alias"), pr.ID) }); err != nil {
		return err
	}
	return back(q, pr, "#models", "Alias removed.")
}

// limitInput reads the shared limit form.
func limitInput(q *console.Request, scopeType, scopeID string) (fleet.LimitInput, error) {
	in := fleet.LimitInput{ScopeType: scopeType, ScopeID: scopeID, ModelID: q.Form("model"), Kind: q.Form("kind")}
	var err error
	if in.Value, err = int64Field(q, "value", fleet.IsCost(in.Kind)); err != nil {
		return in, err
	}
	if q.Form("boost_value") != "" {
		b, err := int64Field(q, "boost_value", fleet.IsCost(in.Kind))
		if err != nil {
			return in, err
		}
		in.BoostValue = &b
		if in.BoostUntil, err = localTime(q, "boost_until"); err != nil {
			return in, err
		}
	}
	return in, nil
}

func (p *pages) projectLimitSet(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectLimits, pr.ID); err != nil {
		return err
	}
	scope := "project"
	if q.Form("scope") == "member" {
		scope = "member"
	}
	in, err := limitInput(q, scope, pr.ID)
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetLimit(q.Ctx(), in) }); err != nil {
		return err
	}
	return back(q, pr, "#limits", "Limit saved.")
}

func (p *pages) projectLimitRemove(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectLimits, pr.ID); err != nil {
		return err
	}
	err := p.do(q, func(tx *fleet.Tx) error {
		return tx.RemoveLimit(q.Ctx(), q.R.PathValue("id"), []string{"project", "member"}, pr.ID)
	})
	if err != nil {
		return err
	}
	return back(q, pr, "#limits", "Limit removed.")
}

func (p *pages) accountCreate(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectKeys, pr.ID); err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error {
		_, err := tx.CreateServiceAccount(q.Ctx(), pr.ID, q.Form("name"), q.Form("description"))
		return err
	}); err != nil {
		return err
	}
	return back(q, pr, "#keys", "Service account created.")
}

func (p *pages) accountStatus(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectKeys, pr.ID); err != nil {
		return err
	}
	status := q.Form("status")
	if err := p.do(q, func(tx *fleet.Tx) error {
		return tx.SetServiceAccountStatus(q.Ctx(), pr.ID, q.R.PathValue("id"), status)
	}); err != nil {
		return err
	}
	return back(q, pr, "#keys", "Service account "+status+".")
}

func (p *pages) accountKey(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectKeys, pr.ID); err != nil {
		return err
	}
	spec := keys.Spec{Name: q.Form("name"), Priority: q.Form("priority"), Classification: q.Form("classification"), CIDRs: q.Lines("cidrs")}
	if days, err := q.Int("expires_days", 0); err != nil {
		return err
	} else if days > 0 {
		t := time.Now().AddDate(0, 0, days)
		spec.ExpiresAt = &t
	}
	var tok string
	if err := p.do(q, func(tx *fleet.Tx) (err error) {
		tok, _, err = tx.IssueServiceKey(q.Ctx(), pr.ID, q.R.PathValue("id"), spec)
		return err
	}); err != nil {
		return err
	}
	q.Render("fleet/secret", "API key", secret{Label: "Use this key as the bearer token for the service account.", Value: tok,
		Back: "/console/projects/" + pr.Slug + "#keys"})
	return nil
}

func (p *pages) keyRevoke(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectKeys, pr.ID); err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.RevokeProjectKey(q.Ctx(), pr.ID, q.R.PathValue("id")) }); err != nil {
		return err
	}
	return back(q, pr, "#keys", "Key revoked.")
}

func (p *pages) keyRotate(q *console.Request, pr fleet.Project) error {
	if err := need(q, authz.ProjectKeys, pr.ID); err != nil {
		return err
	}
	hours, err := q.Int("grace_hours", 24)
	if err != nil {
		return err
	}
	var tok string
	if err := p.do(q, func(tx *fleet.Tx) (err error) {
		tok, _, err = tx.RotateProjectKey(q.Ctx(), pr.ID, q.R.PathValue("id"), time.Duration(hours)*time.Hour)
		return err
	}); err != nil {
		return err
	}
	q.Render("fleet/secret", "Rotated API key", secret{Label: "The old key keeps working during the grace period you chose.", Value: tok,
		Back: "/console/projects/" + pr.Slug + "#keys"})
	return nil
}
