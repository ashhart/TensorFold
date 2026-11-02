package identitypages

import (
	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

type accessPage struct {
	Bindings    []people.Binding
	Roles       []roleOption
	Groups      []people.GroupRow
	Owners      int
	CanOwners   bool
	RoleLabels  map[string]string
	CurrentUser string
}

func (p pages) access() {
	c := p.c
	c.Handle("GET /console/access", authz.RolesWrite, func(q *console.Request) error {
		ctx := q.Ctx()
		bs, err := people.OrgBindings(ctx, c.Pool)
		if err != nil {
			return err
		}
		groups, err := people.ListGroups(ctx, c.Pool, "")
		if err != nil {
			return err
		}
		owners, err := people.ActiveOwners(ctx, c.Pool)
		if err != nil {
			return err
		}
		q.Render("identity/access", "Administrators", accessPage{Bindings: bs, Roles: roleOptions(q, false), Groups: groups, Owners: owners,
			CanOwners: q.Can(authz.OwnersWrite, ""), RoleLabels: roleLabels, CurrentUser: q.User.ID})
		return nil
	})
	c.Handle("POST /console/access", authz.RolesWrite, func(q *console.Request) error {
		role := q.Form("role")
		if err := requireOwnerRights(q, role == "owner"); err != nil {
			return err
		}
		subjectType, subjectID := "group", q.Form("group")
		if email := q.Form("email"); email != "" {
			u, err := directory.FindByEmail(q.Ctx(), c.Pool, email)
			if err != nil {
				return console.Invalid("No active user has the email " + email + ". They must sign in once, or be provisioned over SCIM, first.")
			}
			subjectType, subjectID = "user", u.ID
		} else if subjectID == "" {
			return console.Invalid("Give a user's email or choose a group.")
		}
		if _, err := people.Grant(q.Ctx(), c.Pool, q.Audit("", "", "", nil), role, subjectType, subjectID, ""); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/access", roleLabels[role]+" role granted.")
	})
	c.Handle("POST /console/access/{id}/delete", authz.RolesWrite, func(q *console.Request) error {
		b, err := people.GetBinding(q.Ctx(), c.Pool, q.R.PathValue("id"))
		if err != nil || !b.Org() {
			return friendly(people.ErrNotFound)
		}
		if err := requireOwnerRights(q, b.Role == "owner"); err != nil {
			return err
		}
		if err := people.Revoke(q.Ctx(), c.Pool, q.Audit("", "", "", nil), b.ID); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/access", "Role removed.")
	})
}
