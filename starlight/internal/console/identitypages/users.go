package identitypages

import (
	"net/url"
	"strconv"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

const pageSize = 50

type usersPage struct {
	Rows   []people.UserRow
	Total  int
	Query  string
	Status string
	Page   int
	Prev   string
	Next   string
}

type userPage struct {
	people.UserDetail
	CanWrite bool
	Self     bool
}

func (p pages) users() {
	c := p.c
	c.Handle("GET /console/users", authz.UsersRead, func(q *console.Request) error {
		v := q.R.URL.Query()
		page := max(atoi(v.Get("page")), 1)
		status := v.Get("status")
		if status != "" && status != directory.Active && status != directory.Suspended && status != directory.Deactivated {
			status = ""
		}
		rows, total, err := people.ListUsers(q.Ctx(), c.Pool, people.Filter{Query: v.Get("q"), Status: status, Offset: (page - 1) * pageSize, Limit: pageSize})
		if err != nil {
			return err
		}
		data := usersPage{Rows: rows, Total: total, Query: v.Get("q"), Status: status, Page: page}
		link := func(n int) string {
			return "/console/users?" + url.Values{"q": {data.Query}, "status": {status}, "page": {strconv.Itoa(n)}}.Encode()
		}
		if page > 1 {
			data.Prev = link(page - 1)
		}
		if page*pageSize < total {
			data.Next = link(page + 1)
		}
		q.Render("identity/users", "Users", data)
		return nil
	})
	c.Handle("GET /console/users/{id}", authz.UsersRead, func(q *console.Request) error {
		d, err := people.GetUser(q.Ctx(), c.Pool, q.R.PathValue("id"))
		if err != nil {
			return friendly(err)
		}
		canWrite := q.Can(authz.UsersWrite, "") && (!d.Owner || q.Can(authz.OwnersWrite, ""))
		q.Render("identity/user", d.User.Email, userPage{UserDetail: d, CanWrite: canWrite, Self: d.User.ID == q.User.ID})
		return nil
	})
	c.Handle("POST /console/users/{id}/status", authz.UsersWrite, func(q *console.Request) error {
		id := q.R.PathValue("id")
		if err := p.ownerGate(q, id); err != nil {
			return err
		}
		status := q.Form("status")
		switch status {
		case directory.Active, directory.Suspended:
		case directory.Deactivated:
			if !q.Bool("confirm") {
				return console.Invalid("Tick the confirmation box to deactivate this user.")
			}
		default:
			return console.Invalid("Choose active, suspended or deactivated.")
		}
		if err := people.SetStatus(q.Ctx(), c.Pool, q.Audit("", "", "", nil), id, status); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/users/"+id, "User is now "+status+".")
	})
	c.Handle("POST /console/users/{id}/revoke", authz.UsersWrite, func(q *console.Request) error {
		id := q.R.PathValue("id")
		if err := p.ownerGate(q, id); err != nil {
			return err
		}
		if err := people.RevokeAll(q.Ctx(), c.Pool, q.Audit("", "", "", nil), id); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/users/"+id, "Every session, token, key and task of this user was revoked.")
	})
}

// ownerGate refuses changes to an owner by administrators who cannot manage owners, and to the acting user.
func (p pages) ownerGate(q *console.Request, userID string) error {
	if userID == q.User.ID {
		return console.Invalid("You cannot change your own account here; use My account to sign out devices.")
	}
	owner, err := people.IsOwner(q.Ctx(), p.c.Pool, userID)
	if err != nil {
		return err
	}
	return requireOwnerRights(q, owner)
}
