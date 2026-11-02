package declare

import (
	"fmt"
	"maps"
	"slices"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
)

func (a *applier) projects(f *File) error {
	for _, p := range f.Projects {
		if err := a.project(p); err != nil {
			return fmt.Errorf("project %s: %w", p.Slug, err)
		}
	}
	return nil
}

func (a *applier) project(p Project) error {
	in, err := fleet.ProjectInput{Slug: p.Slug, Name: p.Name, Description: p.Description, Classification: p.Classification,
		MaxPriority: p.MaxPriority, Everyone: p.Everyone, PlanApproval: p.PlanApproval, SecretsGuard: p.SecretsGuard, IsolateCache: p.IsolateCache}.Normalize()
	if err != nil {
		return err
	}
	cur, err := fleet.ProjectBySlug(a.ctx, a.tx.Q(), p.Slug)
	switch {
	case notFound(err):
		if cur, err = a.tx.CreateProject(a.ctx, in); err != nil {
			return err
		}
		a.create("project", p.Slug, in.Classification)
	case err != nil:
		return err
	default:
		var d diff
		field(&d, "name", cur.Name, in.Name)
		field(&d, "description", cur.Description, in.Description)
		field(&d, "classification", cur.Classification, in.Classification)
		field(&d, "max_priority", cur.MaxPriority, in.MaxPriority)
		field(&d, "everyone", cur.Everyone, in.Everyone)
		field(&d, "plan_approval", cur.PlanApproval, in.PlanApproval)
		field(&d, "secrets_guard", cur.SecretsGuard, in.SecretsGuard)
		field(&d, "isolate_prompt_cache", cur.IsolateCache, in.IsolateCache)
		if len(d) > 0 {
			if cur, err = a.tx.UpdateProject(a.ctx, cur.ID, in); err != nil {
				return err
			}
			a.update("project", p.Slug, d...)
		}
	}
	if (cur.ArchivedAt != nil) != p.Archived {
		if err := a.tx.SetArchived(a.ctx, cur.ID, p.Archived); err != nil {
			return err
		}
		a.update("project", p.Slug, fmt.Sprintf("archived %t → %t", cur.ArchivedAt != nil, p.Archived))
	}
	for _, step := range []func(fleet.Project, Project) error{a.serviceAccounts, a.members, a.projectGrants, a.projectAliases, a.projectLimits} {
		if err := step(cur, p); err != nil {
			return err
		}
	}
	return nil
}

func (a *applier) serviceAccounts(pr fleet.Project, p Project) error {
	current, err := fleet.ServiceAccounts(a.ctx, a.tx.Q(), pr.ID)
	if err != nil {
		return err
	}
	for _, sa := range p.ServiceAccounts {
		if slices.ContainsFunc(current, func(c fleet.ServiceAccount) bool { return c.Name == sa.Name }) {
			continue
		}
		if _, err := a.tx.CreateServiceAccount(a.ctx, pr.ID, sa.Name, sa.Description); err != nil {
			return fmt.Errorf("service account %s: %w", sa.Name, err)
		}
		a.create("service account", p.Slug+"/"+sa.Name)
	}
	return nil
}

func (a *applier) members(pr fleet.Project, p Project) error {
	for _, r := range p.Members {
		if err := a.bind(r, pr); err != nil {
			return err
		}
	}
	return nil
}

func (a *applier) projectGrants(pr fleet.Project, p Project) error {
	if len(p.Models) == 0 {
		return nil
	}
	current, err := fleet.Grants(a.ctx, a.tx.Q(), pr.ID, "")
	if err != nil {
		return err
	}
	for _, g := range p.Models {
		m, err := a.modelNamed(g.Model)
		if err != nil {
			return err
		}
		i := slices.IndexFunc(current, func(c fleet.Grant) bool { return c.ModelID == m.ID })
		name := m.Name + " to " + p.Slug
		var d diff
		if i >= 0 {
			fieldCap(&d, "max_context", current[i].MaxContext, g.MaxContext)
			fieldCap(&d, "max_output_tokens", current[i].MaxOutputTokens, g.MaxOutputTokens)
			if len(d) == 0 {
				continue
			}
		}
		if err := a.tx.SetGrant(a.ctx, m.ID, pr.ID, g.MaxContext, g.MaxOutputTokens); err != nil {
			return err
		}
		if i >= 0 {
			a.update("grant", name, d...)
		} else {
			a.create("grant", name)
		}
	}
	return nil
}

func (a *applier) projectAliases(pr fleet.Project, p Project) error {
	if len(p.Aliases) == 0 {
		return nil
	}
	current, err := fleet.Aliases(a.ctx, a.tx.Q(), pr.ID)
	if err != nil {
		return err
	}
	byAlias := map[string]fleet.Alias{}
	for _, al := range current {
		byAlias[al.Alias] = al
	}
	for _, alias := range slices.Sorted(maps.Keys(p.Aliases)) {
		m, err := a.modelNamed(p.Aliases[alias])
		if err != nil {
			return err
		}
		if err := a.alias(byAlias, alias, pr.ID, m, "alias in "+p.Slug); err != nil {
			return err
		}
	}
	return nil
}

func (a *applier) projectLimits(pr fleet.Project, p Project) error {
	for _, l := range p.Limits {
		scope := l.Scope
		switch scope {
		case "":
			scope = "project"
		case "project", "member":
		default:
			return fmt.Errorf("limit %s: scope must be project or member", l.Kind)
		}
		if err := a.limit(l, scope, pr.ID, p.Slug); err != nil {
			return err
		}
	}
	return nil
}

func (a *applier) modelNamed(name string) (fleet.Model, error) {
	m, err := fleet.ModelByName(a.ctx, a.tx.Q(), name)
	if notFound(err) {
		return m, fmt.Errorf("no model named %s; declare it under models", name)
	}
	return m, err
}
