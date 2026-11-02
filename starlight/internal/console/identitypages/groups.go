package identitypages

import (
	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

type groupsPage struct {
	Rows  []people.GroupRow
	Query string
}

type roleOption struct {
	Value string
	Label string
}

type groupPage struct {
	people.GroupDetail
	CanBind   bool
	OrgRoles  []roleOption
	ProjRoles []roleOption
	Projects  []people.ProjectRef
}

var roleLabels = map[string]string{
	"owner": "Owner", "security_admin": "Security administrator", "fleet_operator": "Fleet operator", "model_publisher": "Model publisher",
	"budget_admin": "Budget administrator", "auditor": "Auditor", "project_admin": "Project administrator", "member": "Member", "viewer": "Viewer",
}

// roleOptions lists roles at one scope, leaving out owner for administrators who cannot grant it.
func roleOptions(q *console.Request, projectScoped bool) []roleOption {
	var out []roleOption
	for _, r := range authz.Roles(projectScoped) {
		if r == "owner" && !q.Can(authz.OwnersWrite, "") {
			continue
		}
		out = append(out, roleOption{Value: r, Label: roleLabels[r]})
	}
	return out
}

func (p pages) groups() {
	c := p.c
	c.Handle("GET /console/groups", authz.UsersRead, func(q *console.Request) error {
		query := q.R.URL.Query().Get("q")
		rows, err := people.ListGroups(q.Ctx(), c.Pool, query)
		if err != nil {
			return err
		}
		q.Render("identity/groups", "Groups", groupsPage{Rows: rows, Query: query})
		return nil
	})
	c.Handle("GET /console/groups/{id}", authz.UsersRead, func(q *console.Request) error {
		d, err := people.GetGroup(q.Ctx(), c.Pool, q.R.PathValue("id"))
		if err != nil {
			return friendly(err)
		}
		page := groupPage{GroupDetail: d, CanBind: q.Can(authz.RolesWrite, "")}
		if page.CanBind {
			page.OrgRoles, page.ProjRoles = roleOptions(q, false), roleOptions(q, true)
			if page.Projects, err = people.Projects(q.Ctx(), c.Pool); err != nil {
				return err
			}
		}
		q.Render("identity/group", d.Group.DisplayName, page)
		return nil
	})
	c.Handle("POST /console/groups/{id}/bindings", authz.RolesWrite, func(q *console.Request) error {
		id, role := q.R.PathValue("id"), q.Form("role")
		if err := requireOwnerRights(q, role == "owner"); err != nil {
			return err
		}
		if _, err := people.Grant(q.Ctx(), c.Pool, q.Audit("", "", "", nil), role, "group", id, q.Form("project")); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/groups/"+id, "Role granted to every member of the group.")
	})
	c.Handle("POST /console/groups/{id}/bindings/{binding}/delete", authz.RolesWrite, func(q *console.Request) error {
		id := q.R.PathValue("id")
		b, err := people.GetBinding(q.Ctx(), c.Pool, q.R.PathValue("binding"))
		if err != nil || b.SubjectType != "group" || b.SubjectID != id {
			return friendly(people.ErrNotFound)
		}
		if err := requireOwnerRights(q, b.Role == "owner"); err != nil {
			return err
		}
		if err := people.Revoke(q.Ctx(), c.Pool, q.Audit("", "", "", nil), b.ID); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/groups/"+id, "Role removed from the group.")
	})
}
