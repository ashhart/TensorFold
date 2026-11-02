package identitypages

import (
	"net/url"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/presets"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
)

type ssoRow struct {
	connections.Connection
	PresetLabel string
	Tokens      int
}

type choice struct {
	Value string
	Label string
}

type ssoPage struct {
	Rows     []ssoRow
	Choices  []choice
	CanWrite bool
}

type field struct {
	Name        string
	Label       string
	Help        string
	Kind        string
	Required    bool
	Placeholder string
	Options     []string
	Value       string
}

type newPage struct {
	Preset      string
	PresetLabel string
	Summary     string
	Protocol    string
	Doc         string
	Fields      []field
	URLs        urls
}

type urls struct {
	Redirect    string
	Backchannel string
	ACS         string
	EntityID    string
	Metadata    string
	SLO         string
	SCIM        string
	TestSignIn  string
}

type createdPage struct {
	Conn connections.Connection
	URLs urls
	Cert *clientCert
}

func (p pages) urlsFor(slug string) urls {
	base := strings.TrimRight(p.c.PublicURL, "/")
	return urls{Redirect: base + "/auth/" + slug + "/callback", Backchannel: base + "/auth/" + slug + "/backchannel-logout",
		ACS: samlsp.ACSURL(base, slug), EntityID: samlsp.EntityID(base, slug), Metadata: samlsp.EntityID(base, slug), SLO: samlsp.SLOURL(base, slug),
		SCIM: base + "/scim/v2", TestSignIn: "/auth/" + slug + "/start?return_to=" + url.QueryEscape("/console/sso/"+slug)}
}

func label(name string) string {
	if pr, ok := presets.Get(name); ok {
		return pr.Label
	}
	return name
}

func (p pages) sso() {
	c := p.c
	c.Handle("GET /console/sso", authz.SSORead, p.ssoList)
	c.Handle("GET /console/sso/new", authz.SSOWrite, func(q *console.Request) error {
		name, protocol, _ := strings.Cut(q.R.URL.Query().Get("choice"), ":")
		pr, ok := presets.Get(name)
		if !ok || !pr.Supports(protocol) {
			return q.Redirect("/console/sso", "Choose an identity provider first.")
		}
		q.Render("identity/sso_new", "Add "+pr.Label, p.newForm(pr, protocol, presets.Values{"jit": "on"}))
		return nil
	})
	c.Handle("POST /console/sso", authz.SSOWrite, p.ssoCreate)
	p.ssoDetail()
	p.ssoCert()
}

func (p pages) ssoList(q *console.Request) error {
	ctx := q.Ctx()
	all, err := p.c.Login.Connections().List(ctx)
	if err != nil {
		return err
	}
	counts := map[string]int{}
	rows, err := p.c.Pool.Query(ctx, `SELECT connection_id, count(*) FROM scim_tokens
		WHERE revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now()) GROUP BY connection_id`)
	if err != nil {
		return err
	}
	for rows.Next() {
		var id string
		var n int
		if err := rows.Scan(&id, &n); err != nil {
			rows.Close()
			return err
		}
		counts[id] = n
	}
	rows.Close()
	page := ssoPage{CanWrite: q.Can(authz.SSOWrite, "")}
	for _, cn := range all {
		page.Rows = append(page.Rows, ssoRow{Connection: cn, PresetLabel: label(cn.Preset), Tokens: counts[cn.ID]})
	}
	for _, pr := range presets.All() {
		for _, proto := range pr.Protocols {
			page.Choices = append(page.Choices, choice{Value: pr.Name + ":" + proto, Label: pr.Label + " (" + strings.ToUpper(proto) + ")"})
		}
	}
	q.Render("identity/sso", "Single sign-on", page)
	return nil
}

func (p pages) newForm(pr presets.Preset, protocol string, v presets.Values) newPage {
	page := newPage{Preset: pr.Name, PresetLabel: pr.Label, Summary: pr.Summary, Protocol: protocol, Doc: pr.Doc, URLs: p.urlsFor("<slug>")}
	for _, f := range pr.FieldsFor(protocol) {
		ff := field{Name: f.Name, Label: f.Label, Help: f.Help, Kind: f.Kind, Required: f.Required, Placeholder: f.Placeholder, Options: f.Options}
		if f.Kind != presets.KindSecret && !f.WriteOnly {
			ff.Value = v[f.Name]
		}
		page.Fields = append(page.Fields, ff)
	}
	return page
}

func (p pages) ssoCreate(q *console.Request) error {
	pr, ok := presets.Get(q.Form("preset"))
	protocol := q.Form("protocol")
	if !ok || !pr.Supports(protocol) {
		return q.Redirect("/console/sso", "Choose an identity provider first.")
	}
	v := presets.FromForm(q.R.PostForm)
	in, err := pr.Build(protocol, v)
	if err == nil {
		var cn connections.Connection
		if cn, err = p.c.Login.Connections().Create(q.Ctx(), in, p.recorder(q, "sso.connection_created", map[string]any{"preset": pr.Name, "protocol": protocol})); err == nil {
			p.c.Login.Forget(cn.ID)
			q.Render("identity/sso_created", "Connection saved", createdPage{Conn: cn, URLs: p.urlsFor(cn.Slug), Cert: certFor(cn)})
			return nil
		}
	}
	return &console.Rerender{Page: "identity/sso_new", Title: "Add " + pr.Label, Data: p.newForm(pr, protocol, v), Msg: friendly(err).Error()}
}

// recorder audits a connection change inside the store's transaction, naming the connection as it was written.
func (p pages) recorder(q *console.Request, action string, details map[string]any) connections.Recorder {
	return func(tx pgx.Tx, cn connections.Connection) error {
		d := map[string]any{"slug": cn.Slug, "name": cn.Name}
		for k, v := range details {
			d[k] = v
		}
		return audit.Append(q.Ctx(), tx, q.Audit(action, "idp_connection", cn.ID, d))
	}
}
