package identitypages

import (
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

type keyRow struct {
	keys.Key
	Project string
	Live    bool
}

type accountPage struct {
	User          directory.User
	Groups        []directory.Group
	Bindings      []people.Binding
	KeyProjects   []people.ProjectRef
	Keys          []keyRow
	Devices       []people.Device
	Sessions      []people.Session
	Current       string
	Tasks         []people.Task
	NewSecret     string
	NewSecretName string
}

func (p pages) account() {
	c := p.c
	c.Handle("GET /console/account", "", func(q *console.Request) error { return p.renderAccount(q, "", "") })
	c.Handle("POST /console/account/keys", "", p.issueKey)
	c.Handle("POST /console/account/keys/{id}/rotate", "", func(q *console.Request) error {
		hours, err := q.Int("grace_hours", 24)
		if err != nil || hours < 0 || hours > 168 {
			return console.Invalid("Keep the old key working for 0 to 168 hours.")
		}
		tok, k, err := people.RotateOwnKey(q.Ctx(), c.Pool, q.Audit("", "", "", nil), q.User.ID, q.R.PathValue("id"), time.Duration(hours)*time.Hour)
		if err != nil {
			return asInvalid(err)
		}
		return p.renderAccount(q, tok, k.Name)
	})
	c.Handle("POST /console/account/keys/{id}/revoke", "", func(q *console.Request) error {
		if err := people.RevokeOwnKey(q.Ctx(), c.Pool, q.Audit("", "", "", nil), q.User.ID, q.R.PathValue("id")); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/account", "Key revoked.")
	})
	c.Handle("POST /console/account/devices/{id}/revoke", "", func(q *console.Request) error {
		if err := people.RevokeDevice(q.Ctx(), c.Pool, q.Audit("", "", "", nil), q.User.ID, q.R.PathValue("id")); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/account", "Device signed out.")
	})
	c.Handle("POST /console/account/sessions/{id}/revoke", "", func(q *console.Request) error {
		if err := people.RevokeSession(q.Ctx(), c.Pool, q.Audit("", "", "", nil), q.User.ID, q.R.PathValue("id")); err != nil {
			return friendly(err)
		}
		return q.Redirect("/console/account", "Session signed out.")
	})
	c.Handle("POST /console/account/tasks/{id}/revoke", "", func(q *console.Request) error {
		if err := people.RevokeTask(q.Ctx(), c.Pool, q.Audit("", "", "", nil), q.User.ID, q.R.PathValue("id")); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/account", "Task and its subtasks revoked.")
	})
}

func (p pages) issueKey(q *console.Request) error {
	project := q.Form("project")
	if !q.Can(authz.OwnKeys, project) {
		return console.Invalid("You cannot create keys in that project.")
	}
	days, err := q.Int("expires_days", 90)
	if err != nil || days < 0 || days > 730 {
		return console.Invalid("Choose an expiry of 0 (never) to 730 days.")
	}
	spec := keys.Spec{Name: q.Form("name"), OwnerID: q.User.ID, ProjectID: project, Priority: q.Form("priority"), CIDRs: q.Lines("cidrs")}
	if days > 0 {
		exp := time.Now().Add(time.Duration(days) * 24 * time.Hour)
		spec.ExpiresAt = &exp
	}
	tok, k, err := people.IssueOwnKey(q.Ctx(), p.c.Pool, q.Audit("", "", "", nil), spec)
	if err != nil {
		return asInvalid(err)
	}
	return p.renderAccount(q, tok, k.Name)
}

func (p pages) renderAccount(q *console.Request, secret, secretName string) error {
	ctx, pool := q.Ctx(), p.c.Pool
	page := accountPage{User: q.User, Current: q.Session.ID, NewSecret: secret, NewSecretName: secretName}
	var err error
	if page.Groups, err = directory.UserGroups(ctx, pool, q.User.ID); err != nil {
		return err
	}
	if page.Bindings, err = people.UserBindings(ctx, pool, q.User.ID); err != nil {
		return err
	}
	projects, err := people.Projects(ctx, pool)
	if err != nil {
		return err
	}
	slugs := map[string]string{}
	for _, pr := range projects {
		slugs[pr.ID] = pr.Slug
		if q.Can(authz.OwnKeys, pr.ID) {
			page.KeyProjects = append(page.KeyProjects, pr)
		}
	}
	mine, err := keys.List(ctx, pool, keys.Filter{OwnerType: "user", OwnerID: q.User.ID})
	if err != nil {
		return err
	}
	now := time.Now()
	for _, k := range mine {
		page.Keys = append(page.Keys, keyRow{Key: k, Project: slugs[k.ProjectID], Live: k.Live(now)})
	}
	if page.Devices, err = people.Devices(ctx, pool, q.User.ID); err != nil {
		return err
	}
	if page.Sessions, err = people.Sessions(ctx, pool, q.User.ID); err != nil {
		return err
	}
	if page.Tasks, err = people.Tasks(ctx, pool, q.User.ID); err != nil {
		return err
	}
	q.Render("identity/account", "My account", page)
	return nil
}
