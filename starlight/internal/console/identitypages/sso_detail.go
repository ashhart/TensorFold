package identitypages

import (
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/scim"
)

type detailPage struct {
	Conn          connections.Connection
	PresetLabel   string
	URLs          urls
	Cert          *clientCert
	Tokens        []scim.TokenInfo
	NewToken      string
	CanWrite      bool
	Form          editForm
	SPCertificate string
	Strategies    []string
	AuthMethods   []string
	NameIDFormats []string
	Now           time.Time
}

func (p pages) ssoDetail() {
	c := p.c
	c.Handle("GET /console/sso/{slug}", authz.SSORead, func(q *console.Request) error { return p.renderDetail(q, "") })
	c.Handle("POST /console/sso/{slug}", authz.SSOWrite, p.ssoUpdate)
	c.Handle("POST /console/sso/{slug}/enabled", authz.SSOWrite, func(q *console.Request) error {
		cn, err := p.conn(q)
		if err != nil {
			return err
		}
		on := q.Form("enabled") == "on"
		action, notice := "sso.connection_disabled", "Connection disabled; its users are signed out and cannot sign in."
		if on {
			action, notice = "sso.connection_enabled", "Connection enabled."
		}
		if err := c.Login.Connections().SetEnabled(q.Ctx(), cn.ID, on, p.recorder(q, action, nil)); err != nil {
			return friendly(err)
		}
		c.Login.Forget(cn.ID)
		return q.Redirect("/console/sso/"+cn.Slug, notice)
	})
	c.Handle("POST /console/sso/{slug}/delete", authz.SSOWrite, func(q *console.Request) error {
		cn, err := p.conn(q)
		if err != nil {
			return err
		}
		if q.Form("confirm") != cn.Slug {
			return console.Invalid("Type the connection's slug to confirm deletion.")
		}
		if err := c.Login.Connections().Delete(q.Ctx(), cn.ID, p.recorder(q, "sso.connection_deleted", map[string]any{"preset": cn.Preset})); err != nil {
			return friendly(err)
		}
		c.Login.Forget(cn.ID)
		c.Flush()
		return q.Redirect("/console/sso", "Connection "+cn.Name+" deleted.")
	})
	c.Handle("POST /console/sso/{slug}/scim", authz.SSOWrite, p.scimIssue)
	c.Handle("POST /console/sso/{slug}/scim/{id}/revoke", authz.SSOWrite, func(q *console.Request) error {
		cn, err := p.conn(q)
		if err != nil {
			return err
		}
		id := q.R.PathValue("id")
		err = db.Tx(q.Ctx(), c.Pool, func(tx pgx.Tx) error {
			if err := scim.RevokeToken(q.Ctx(), tx, cn.ID, id); err != nil {
				return err
			}
			return audit.Append(q.Ctx(), tx, q.Audit("scim.token_revoked", "idp_connection", cn.ID, map[string]any{"token_id": id, "slug": cn.Slug}))
		})
		if err != nil {
			return friendly(err)
		}
		return q.Redirect("/console/sso/"+cn.Slug, "Provisioning token revoked.")
	})
}

func (p pages) conn(q *console.Request) (connections.Connection, error) {
	cn, err := p.c.Login.Connections().GetBySlug(q.Ctx(), q.R.PathValue("slug"))
	return cn, friendly(err)
}

func (p pages) detail(q *console.Request, cn connections.Connection, token string) (detailPage, error) {
	page := detailPage{Conn: cn, PresetLabel: label(cn.Preset), URLs: p.urlsFor(cn.Slug), Cert: certFor(cn), NewToken: token,
		CanWrite: q.Can(authz.SSOWrite, ""), Form: formFrom(cn), Strategies: strategies, AuthMethods: authMethods, NameIDFormats: nameIDFormat,
		Now: time.Now()}
	if cn.Config.SAML != nil {
		page.SPCertificate = cn.Config.SAML.SPCertificate
	}
	var err error
	page.Tokens, err = scim.ListTokens(q.Ctx(), p.c.Pool, cn.ID)
	return page, err
}

func (p pages) renderDetail(q *console.Request, token string) error {
	cn, err := p.conn(q)
	if err != nil {
		return err
	}
	page, err := p.detail(q, cn, token)
	if err != nil {
		return err
	}
	q.Render("identity/sso_detail", cn.Name, page)
	return nil
}

func (p pages) ssoUpdate(q *console.Request) error {
	cn, err := p.conn(q)
	if err != nil {
		return err
	}
	in, err := inputFrom(cn, q)
	if err == nil {
		var saved connections.Connection
		details := map[string]any{"preset": in.Preset, "secrets_changed": in.Secrets != (connections.Secrets{})}
		if saved, err = p.c.Login.Connections().Update(q.Ctx(), cn.ID, in, p.recorder(q, "sso.connection_updated", details)); err == nil {
			p.c.Login.Forget(cn.ID)
			return q.Redirect("/console/sso/"+saved.Slug, "Connection saved.")
		}
	}
	page, perr := p.detail(q, cn, "")
	if perr != nil {
		return perr
	}
	attempt := cn
	attempt.Name, attempt.Slug, attempt.Domains, attempt.JIT, attempt.Config = in.Name, in.Slug, in.Domains, in.JIT, in.Config
	page.Form = formFrom(attempt)
	return &console.Rerender{Page: "identity/sso_detail", Title: cn.Name, Data: page, Msg: friendly(err).Error()}
}

func (p pages) scimIssue(q *console.Request) error {
	cn, err := p.conn(q)
	if err != nil {
		return err
	}
	days, err := q.Int("expires_days", 365)
	if err != nil || days < 0 || days > 1095 {
		return console.Invalid("Choose an expiry of 0 (never) to 1095 days.")
	}
	var expires *time.Time
	if days > 0 {
		t := time.Now().Add(time.Duration(days) * 24 * time.Hour)
		expires = &t
	}
	var token string
	err = db.Tx(q.Ctx(), p.c.Pool, func(tx pgx.Tx) error {
		tok, info, err := scim.IssueToken(q.Ctx(), tx, cn.ID, q.Form("label"), expires)
		if err != nil {
			return err
		}
		token = tok
		return audit.Append(q.Ctx(), tx, q.Audit("scim.token_issued", "idp_connection", cn.ID, map[string]any{"token_id": info.ID, "label": info.Label,
			"slug": cn.Slug}))
	})
	if err != nil {
		return asInvalid(err)
	}
	return p.renderDetail(q, token)
}
