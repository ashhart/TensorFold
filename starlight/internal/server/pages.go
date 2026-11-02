package server

import (
	"context"
	"net/http"
	"sort"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oauth"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// oauthPages renders the OAuth flows with the shared web templates.
type oauthPages struct{ r *web.Renderer }

func (p oauthPages) Approve(w http.ResponseWriter, r *http.Request, a oauth.ApprovePage) {
	keys := make([]string, 0, len(a.Hidden))
	for k := range a.Hidden {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	hidden := make([]web.Hidden, 0, len(keys))
	for _, k := range keys {
		hidden = append(hidden, web.Hidden{Name: k, Value: a.Hidden[k]})
	}
	p.r.Render(w, r, http.StatusOK, web.PageApprove, "Allow "+a.ClientName, web.ApprovePage{ClientName: a.ClientName, DeviceName: a.DeviceName,
		UserEmail: a.UserEmail, Scopes: a.Scopes, Action: a.Action, CSRF: a.CSRF, Hidden: hidden})
}

func (p oauthPages) Device(w http.ResponseWriter, r *http.Request, d oauth.DevicePage) {
	action, status := "/device", http.StatusOK
	if d.Confirm {
		action = "/device/decide"
	}
	if d.Error != "" {
		status = http.StatusBadRequest
	}
	p.r.Render(w, r, status, web.PageDevice, "Connect a device", web.DevicePage{Action: action, CSRF: d.CSRF, UserCode: d.UserCode, Error: d.Error,
		Confirm: d.Confirm, ClientName: d.ClientName, DeviceName: d.DeviceName})
}

func (p oauthPages) Message(w http.ResponseWriter, r *http.Request, m oauth.MessagePage) {
	kind := "info"
	switch {
	case m.Status >= 400:
		kind = "error"
	case m.Status == http.StatusOK:
		kind = "success"
	}
	p.r.Message(w, r, m.Status, kind, m.Title, m.Body)
}

// meBuilder answers /oauth/userinfo with the user's projects and the models each may use.
func meBuilder(pool *pgxpool.Pool, cat *catalog.Catalog, org func(context.Context) string) func(context.Context, string) (discovery.Me, error) {
	return func(ctx context.Context, userID string) (discovery.Me, error) {
		u, err := directory.GetUser(ctx, pool, userID)
		if err != nil || u.Status != directory.Active {
			return discovery.Me{}, oauth.ErrInvalidGrant
		}
		groups, err := directory.GroupIDs(ctx, pool, userID)
		if err != nil {
			return discovery.Me{}, err
		}
		g, err := authz.LoadUser(ctx, pool, userID, groups)
		if err != nil {
			return discovery.Me{}, err
		}
		me := discovery.Me{UserID: u.ID, Email: u.Email, DisplayName: u.DisplayName, OrgName: org(ctx), Projects: []discovery.Project{}}
		ids := g.ProjectIDs(authz.Inference)
		rows, err := pool.Query(ctx, "SELECT id, slug, name, classification, everyone FROM projects WHERE id = ANY($1::uuid[]) AND archived_at IS NULL ORDER BY slug", ids)
		if err != nil {
			return me, err
		}
		defer rows.Close()
		for rows.Next() {
			var p discovery.Project
			var everyone bool
			if err := rows.Scan(&p.ID, &p.Slug, &p.Name, &p.Classification, &everyone); err != nil {
				return me, err
			}
			p.Role = g.TopRole(p.ID)
			if p.Role == "" {
				p.Role = "member"
			}
			p.Default = len(ids) == 1 || everyone
			me.Projects = append(me.Projects, p)
		}
		if err := rows.Err(); err != nil {
			return me, err
		}
		markOneDefault(me.Projects)
		for i := range me.Projects {
			models, err := cat.Visible(ctx, me.Projects[i].ID)
			if err != nil {
				return me, err
			}
			for _, m := range models {
				me.Projects[i].Models = append(me.Projects[i].Models, m.Name)
			}
		}
		return me, nil
	}
}

func markOneDefault(ps []discovery.Project) {
	seen := false
	for i := range ps {
		if ps[i].Default && !seen {
			seen = true
			continue
		}
		ps[i].Default = false
	}
	if !seen && len(ps) > 0 {
		ps[0].Default = true
	}
}

// refreshEgress keeps the provider allowlist in step with the providers table and offline mode.
func refreshEgress(ctx context.Context, pool *pgxpool.Pool, e *providers.Egress) {
	for {
		allow := map[string]bool{}
		offline := true
		if err := pool.QueryRow(ctx, "SELECT offline FROM settings WHERE id").Scan(&offline); err != nil {
			offline = true
		}
		rows, err := pool.Query(ctx, "SELECT base_url, internal FROM providers WHERE enabled")
		if err == nil {
			for rows.Next() {
				var base string
				var internal bool
				if rows.Scan(&base, &internal) == nil {
					if h, err := providers.HostOf(base); err == nil {
						allow[h] = allow[h] || internal
					}
				}
			}
			rows.Close()
		}
		e.Allow(allow, offline)
		select {
		case <-ctx.Done():
			return
		case <-time.After(30 * time.Second):
		}
	}
}
