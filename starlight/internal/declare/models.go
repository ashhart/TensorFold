package declare

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

func (a *applier) models(f *File) error {
	for _, m := range f.Models {
		if err := a.model(m); err != nil {
			return fmt.Errorf("model %s: %w", m.Name, err)
		}
	}
	return nil
}

func (a *applier) model(m Model) error {
	in, err := fleet.ModelInput{Name: m.Name, DisplayName: m.DisplayName, Description: m.Description, Kind: m.Kind, RoutePolicy: m.RoutePolicy,
		Status: m.Status, Capabilities: fleet.Capabilities(m.Capabilities), Chargeback: m.Chargeback.fleet()}.Normalize()
	if err != nil {
		return err
	}
	cur, err := fleet.ModelByName(a.ctx, a.tx.Q(), m.Name)
	switch {
	case notFound(err):
		if cur, err = a.tx.CreateModel(a.ctx, in); err != nil {
			return err
		}
		a.create("model", m.Name, in.RoutePolicy)
	case err != nil:
		return err
	default:
		var d diff
		field(&d, "display_name", cur.DisplayName, in.DisplayName)
		field(&d, "description", cur.Description, in.Description)
		field(&d, "kind", cur.Kind, in.Kind)
		field(&d, "route_policy", cur.RoutePolicy, in.RoutePolicy)
		field(&d, "status", cur.Status, in.Status)
		field(&d, "context_window", cur.Capabilities.ContextWindow, in.Capabilities.ContextWindow)
		field(&d, "tools", cur.Capabilities.Tools, in.Capabilities.Tools)
		field(&d, "vision", cur.Capabilities.Vision, in.Capabilities.Vision)
		field(&d, "reasoning", cur.Capabilities.Reasoning, in.Capabilities.Reasoning)
		field(&d, "chargeback.input", fleet.Dollars(cur.Chargeback.Input), fleet.Dollars(in.Chargeback.Input))
		field(&d, "chargeback.output", fleet.Dollars(cur.Chargeback.Output), fleet.Dollars(in.Chargeback.Output))
		field(&d, "chargeback.cached", fleet.Dollars(cur.Chargeback.Cached), fleet.Dollars(in.Chargeback.Cached))
		if len(d) > 0 {
			if cur, err = a.tx.UpdateModel(a.ctx, cur.ID, in); err != nil {
				return err
			}
			a.update("model", m.Name, d...)
		}
	}
	if err := a.everyone(cur, m); err != nil {
		return err
	}
	if err := a.globalAliases(cur, m.Aliases); err != nil {
		return err
	}
	for _, d := range m.Deployments {
		if err := a.deployment(cur, d); err != nil {
			return fmt.Errorf("deployment %s: %w", d.Name, err)
		}
	}
	return a.routes(cur, m.Routes)
}

// everyone grants the model to every project within caps, or withdraws that grant when set to false.
func (a *applier) everyone(m fleet.Model, want Model) error {
	if want.Everyone == nil {
		return nil
	}
	grants, err := fleet.Grants(a.ctx, a.tx.Q(), "", m.ID)
	if err != nil {
		return err
	}
	var org *fleet.Grant
	for i := range grants {
		if grants[i].ProjectID == "" {
			org = &grants[i]
		}
	}
	switch {
	case !*want.Everyone && org != nil:
		if err := a.tx.RemoveGrant(a.ctx, org.ID); err != nil {
			return err
		}
		a.update("model", m.Name, "no longer granted to every project")
	case *want.Everyone && org == nil:
		if err := a.tx.SetGrant(a.ctx, m.ID, "", want.MaxContext, want.MaxOutput); err != nil {
			return err
		}
		a.create("grant", m.Name+" to every project")
	case *want.Everyone:
		var d diff
		fieldCap(&d, "max_context", org.MaxContext, want.MaxContext)
		fieldCap(&d, "max_output_tokens", org.MaxOutputTokens, want.MaxOutput)
		if len(d) > 0 {
			if err := a.tx.SetGrant(a.ctx, m.ID, "", want.MaxContext, want.MaxOutput); err != nil {
				return err
			}
			a.update("grant", m.Name+" to every project", d...)
		}
	}
	return nil
}

func (a *applier) globalAliases(m fleet.Model, aliases []string) error {
	current, err := fleet.Aliases(a.ctx, a.tx.Q(), "global")
	if err != nil {
		return err
	}
	byAlias := map[string]fleet.Alias{}
	for _, al := range current {
		byAlias[al.Alias] = al
	}
	for _, alias := range aliases {
		if err := a.alias(byAlias, alias, "", m, "global alias"); err != nil {
			return err
		}
	}
	return nil
}

// alias points alias at m in one project, or globally when projectID is empty, reporting a create or a move.
func (a *applier) alias(current map[string]fleet.Alias, alias, projectID string, m fleet.Model, kind string) error {
	old, ok := current[alias]
	if ok && old.ModelID == m.ID {
		return nil
	}
	if err := a.tx.SetAlias(a.ctx, alias, projectID, m.ID); err != nil {
		return err
	}
	if ok {
		a.update(kind, alias, fmt.Sprintf("model %q → %q", old.ModelName, m.Name))
	} else {
		a.create(kind, alias, "→ "+m.Name)
	}
	return nil
}

func (a *applier) deployment(m fleet.Model, d Deployment) error {
	minimum := 1
	if d.MinReplicas != nil {
		minimum = *d.MinReplicas
	}
	in, err := fleet.DeploymentInput{Name: d.Name, ModelID: m.ID, Spec: d.Spec.Spec, Pool: d.Pool, Selector: d.Selector, Mode: d.Mode,
		MinReplicas: minimum, MaxReplicas: d.MaxReplicas, IdleSeconds: d.IdleSeconds}.Normalize()
	if err != nil {
		return err
	}
	cur, err := fleet.DeploymentByName(a.ctx, a.tx.Q(), d.Name)
	switch {
	case notFound(err):
		if cur, err = a.tx.CreateDeployment(a.ctx, in); err != nil {
			return err
		}
		a.create("deployment", d.Name, m.Name+" from "+in.Spec.Model.Key())
	case err != nil:
		return err
	case cur.ModelID != m.ID:
		return fmt.Errorf("the deployment already serves model %s; deployment names are unique across models", cur.ModelName)
	default:
		changes := specDiff(cur.Spec, in.Spec)
		field(&changes, "pool", cur.Pool, in.Pool)
		fieldMap(&changes, "selector", cur.Selector, in.Selector)
		field(&changes, "mode", cur.Mode, in.Mode)
		field(&changes, "min_replicas", cur.MinReplicas, in.MinReplicas)
		field(&changes, "max_replicas", cur.MaxReplicas, in.MaxReplicas)
		field(&changes, "idle_seconds", cur.IdleSeconds, in.IdleSeconds)
		if len(changes) > 0 {
			if cur, err = a.tx.UpdateDeployment(a.ctx, cur.ID, in); err != nil {
				return err
			}
			a.update("deployment", d.Name, changes...)
		}
	}
	if d.Status != "" && d.Status != cur.Status {
		if err := a.tx.SetDeploymentStatus(a.ctx, cur.ID, d.Status); err != nil {
			return err
		}
		a.update("deployment", d.Name, fmt.Sprintf("status %q → %q", cur.Status, d.Status))
	}
	return nil
}

// specDiff names each engine setting that differs; any difference starts a new revision.
func specDiff(before, after wire.Spec) diff {
	var b, f map[string]json.RawMessage
	bj, _ := json.Marshal(before)
	fj, _ := json.Marshal(after)
	_ = json.Unmarshal(bj, &b)
	_ = json.Unmarshal(fj, &f)
	keys := slices.Sorted(maps.Keys(b))
	for k := range f {
		if _, ok := b[k]; !ok {
			keys = append(keys, k)
		}
	}
	var d diff
	for _, k := range keys {
		if !bytes.Equal(b[k], f[k]) {
			d = append(d, fmt.Sprintf("spec.%s %s → %s (new revision)", k, raw(b[k]), raw(f[k])))
		}
	}
	return d
}

func raw(v json.RawMessage) string {
	if len(v) == 0 {
		return "unset"
	}
	return string(v)
}

func (a *applier) routes(m fleet.Model, routes []Route) error {
	if len(routes) == 0 {
		return nil
	}
	current, err := fleet.Routes(a.ctx, a.tx.Q(), m.ID)
	if err != nil {
		return err
	}
	for _, r := range routes {
		if err := a.route(m, r, current); err != nil {
			return fmt.Errorf("route to %s: %w", r.Provider, err)
		}
	}
	return nil
}

func (a *applier) route(m fleet.Model, r Route, current []fleet.Route) error {
	pr, err := fleet.ProviderByName(a.ctx, a.tx.Q(), r.Provider)
	if notFound(err) {
		return fmt.Errorf("no provider named %s; declare it under providers", r.Provider)
	}
	if err != nil {
		return err
	}
	in, err := fleet.RouteInput{ProviderID: pr.ID, UpstreamModel: r.UpstreamModel, Protocols: r.Protocols, Priority: r.Priority,
		Prices: r.Prices.fleet(), MaxConcurrency: r.MaxConcurrency, Enabled: r.Enabled == nil || *r.Enabled}.Normalize(pr.Kind)
	if err != nil {
		return err
	}
	name := m.Name + " → " + pr.Name + "/" + in.UpstreamModel
	for _, cur := range current {
		if cur.ProviderID != pr.ID || cur.UpstreamModel != in.UpstreamModel {
			continue
		}
		var d diff
		fieldList(&d, "protocols", cur.Protocols, in.Protocols)
		field(&d, "priority", cur.Priority, in.Priority)
		field(&d, "prices.input", fleet.Dollars(cur.Prices.Input), fleet.Dollars(in.Prices.Input))
		field(&d, "prices.output", fleet.Dollars(cur.Prices.Output), fleet.Dollars(in.Prices.Output))
		field(&d, "prices.cached", fleet.Dollars(cur.Prices.Cached), fleet.Dollars(in.Prices.Cached))
		field(&d, "max_concurrency", cur.MaxConcurrency, in.MaxConcurrency)
		field(&d, "enabled", cur.Enabled, in.Enabled)
		if len(d) == 0 {
			return nil
		}
		if err := a.tx.SetRoute(a.ctx, m.ID, in); err != nil {
			return err
		}
		a.update("route", name, d...)
		return nil
	}
	if err := a.tx.SetRoute(a.ctx, m.ID, in); err != nil {
		return err
	}
	a.create("route", name)
	return nil
}
