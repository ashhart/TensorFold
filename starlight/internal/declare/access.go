package declare

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
)

func (a *applier) orgRoles(f *File) error {
	for _, r := range f.OrgRoles {
		if err := a.bind(r, fleet.Project{}); err != nil {
			return fmt.Errorf("org role %s: %w", r.Role, err)
		}
	}
	return nil
}

// bind grants a role in pr, or organisation-wide when pr has no id; existing bindings are left alone.
func (a *applier) bind(r Role, pr fleet.Project) error {
	st, subjects, label, err := a.subject(r, pr)
	if err != nil {
		return err
	}
	scope := "the organisation"
	if pr.ID != "" {
		scope = pr.Slug
	}
	for _, id := range subjects {
		created, err := a.tx.Bind(a.ctx, r.Role, st, id, pr.ID)
		if err != nil {
			return fmt.Errorf("%s for %s: %w", r.Role, label, err)
		}
		if created {
			a.create("role", r.Role+" for "+label+" in "+scope)
		}
	}
	return nil
}

// subject finds who a binding names: a user by email, an identity provider group or a project service account.
func (a *applier) subject(r Role, pr fleet.Project) (string, []string, string, error) {
	named := 0
	for _, set := range []bool{r.User != "", r.Group != nil, r.ServiceAccount != ""} {
		if set {
			named++
		}
	}
	if named != 1 {
		return "", nil, "", errors.New("name exactly one of user, group or service_account")
	}
	switch {
	case r.User != "":
		u, err := fleet.UserByEmail(a.ctx, a.tx.Q(), r.User)
		return "user", []string{u.ID}, r.User, err
	case r.Group != nil:
		label := r.Group.Connection + "/" + r.Group.ID
		groups, created, err := a.tx.GroupsFor(a.ctx, r.Group.Connection, r.Group.ID)
		if created {
			a.create("group", label, "declared before anyone from it has signed in")
		}
		return "group", groups, "group " + label, err
	case pr.ID == "":
		return "", nil, "", errors.New("service accounts hold project roles only")
	}
	accounts, err := fleet.ServiceAccounts(a.ctx, a.tx.Q(), pr.ID)
	if err != nil {
		return "", nil, "", err
	}
	i := slices.IndexFunc(accounts, func(s fleet.ServiceAccount) bool { return s.Name == r.ServiceAccount })
	if i < 0 {
		return "", nil, "", fmt.Errorf("no service account %s in %s; declare it under service_accounts", r.ServiceAccount, pr.Slug)
	}
	return "service_account", []string{accounts[i].ID}, "service account " + r.ServiceAccount, nil
}

func (a *applier) orgLimits(f *File) error {
	for _, l := range f.Limits {
		if l.Scope != "" && l.Scope != "org" {
			return fmt.Errorf("limit %s: top-level limits are organisation-wide; put project limits under their project", l.Kind)
		}
		if err := a.limit(l, "org", "", ""); err != nil {
			return err
		}
	}
	return nil
}

// limit sets one limit unless the same value and boost are in place; a boost whose end has passed is dropped.
func (a *applier) limit(l Limit, scopeType, scopeID, where string) error {
	in := fleet.LimitInput{ScopeType: scopeType, ScopeID: scopeID, Kind: l.Kind}
	label := l.Kind
	if l.Model != "" {
		m, err := a.modelNamed(l.Model)
		if err != nil {
			return err
		}
		in.ModelID, label = m.ID, label+" for "+m.Name
	}
	switch {
	case scopeType == "member":
		label += " per member of " + where
	case where != "":
		label += " in " + where
	}
	var err error
	if in.Value, err = amount(l.Kind, l.Value); err != nil {
		return fmt.Errorf("limit %s: %w", label, err)
	}
	if l.BoostValue != "" || l.BoostUntil != "" {
		until, err := time.Parse(time.RFC3339, l.BoostUntil)
		if err != nil {
			return fmt.Errorf("limit %s: boost_until must be an RFC 3339 time such as 2026-12-01T18:00:00Z", label)
		}
		v, err := amount(l.Kind, l.BoostValue)
		if err != nil {
			return fmt.Errorf("limit %s: boost: %w", label, err)
		}
		if until.After(time.Now()) {
			in.BoostValue, in.BoostUntil = &v, &until
		}
	}
	current, err := fleet.Limits(a.ctx, a.tx.Q(), []string{scopeType}, scopeID)
	if err != nil {
		return err
	}
	shown := fleet.Limit{Kind: in.Kind}
	i := slices.IndexFunc(current, func(c fleet.Limit) bool { return c.Kind == in.Kind && c.ModelID == in.ModelID })
	var d diff
	if i >= 0 {
		c := current[i]
		field(&d, "value", c.Amount(c.Value), shown.Amount(in.Value))
		field(&d, "boost", boostText(c, c.BoostValue, c.BoostUntil), boostText(c, in.BoostValue, in.BoostUntil))
		if len(d) == 0 {
			return nil
		}
	}
	if err := a.tx.SetLimit(a.ctx, in); err != nil {
		return fmt.Errorf("limit %s: %w", label, err)
	}
	if i >= 0 {
		a.update("limit", label, d...)
	} else {
		a.create("limit", label, shown.Amount(in.Value))
	}
	return nil
}

// amount reads a limit value: dollars for cost kinds, a whole number otherwise.
func amount(kind, s string) (int64, error) {
	if fleet.IsCost(kind) {
		return fleet.ParseDollars(kind, s)
	}
	n, err := strconv.ParseInt(strings.ReplaceAll(strings.TrimSpace(s), "_", ""), 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("%s must be a non-negative whole number", kind)
	}
	return n, nil
}

func boostText(l fleet.Limit, v *int64, until *time.Time) string {
	if v == nil || until == nil {
		return "none"
	}
	return l.Amount(*v) + " until " + until.UTC().Format(time.RFC3339)
}
