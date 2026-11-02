package login

import (
	"errors"
	"net/http"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/presets"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// setupPage opens the first-owner wizard from the link `starlight init` prints.
func (h *Handler) setupPage(w http.ResponseWriter, r *http.Request) {
	code := r.URL.Query().Get("code")
	if _, err := setupCode(r.Context(), h.cfg.Pool, code); err != nil {
		e := explain(errSetupDone)
		h.fail(w, r, e.status, e.heading, e.body)
		return
	}
	h.renderSetup(w, r, code, "", "")
}

func (h *Handler) renderSetup(w http.ResponseWriter, r *http.Request, code, notice, problem string) {
	page := web.SetupPage{Code: code, PublicURL: h.cfg.PublicURL, Notice: notice, Error: problem}
	all, err := h.conns.List(r.Context())
	if err != nil {
		h.fail(w, r, http.StatusServiceUnavailable, "Setup unavailable", "Starlight cannot read its connections right now.")
		return
	}
	for _, c := range all {
		if c.Enabled {
			page.Connections = append(page.Connections, h.setupConnection(c))
		}
	}
	page.Step = "signin"
	if len(page.Connections) == 0 {
		page.Step = "choose"
		page.Presets = choices()
	}
	h.cfg.Web.Render(w, r, http.StatusOK, web.PageSetup, "Set up single sign-on", page)
}

func (h *Handler) setupConnection(c connections.Connection) web.SetupConnection {
	return web.SetupConnection{Slug: c.Slug, Name: c.Name, Protocol: c.Protocol, RedirectURI: h.redirectURI(c.Slug),
		ACSURL: samlsp.ACSURL(h.cfg.PublicURL, c.Slug), EntityID: samlsp.EntityID(h.cfg.PublicURL, c.Slug),
		MetadataURL: samlsp.EntityID(h.cfg.PublicURL, c.Slug)}
}

func choices() []web.PresetChoice {
	var out []web.PresetChoice
	for _, p := range presets.All() {
		out = append(out, web.PresetChoice{Name: p.Name, Label: p.Label, Protocols: p.Protocols})
	}
	return out
}

// setupPost handles the wizard's three actions; every one re-checks the setup code.
func (h *Handler) setupPost(w http.ResponseWriter, r *http.Request) {
	if !session.SameOrigin(r) || !h.limiter.allow(r) {
		h.fail(w, r, http.StatusForbidden, "Request refused", "Start again from the setup link.")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
	if err := r.ParseForm(); err != nil {
		h.fail(w, r, http.StatusBadRequest, "Request refused", "The form could not be read.")
		return
	}
	code := r.PostForm.Get("code")
	codeID, err := setupCode(r.Context(), h.cfg.Pool, code)
	if err != nil {
		e := explain(errSetupDone)
		h.fail(w, r, e.status, e.heading, e.body)
		return
	}
	switch r.PostForm.Get("action") {
	case "choose":
		h.setupChoose(w, r, code)
	case "create":
		h.setupCreate(w, r, code)
	case "start":
		conn, err := h.conns.GetBySlug(r.Context(), r.PostForm.Get("slug"))
		if err != nil || !conn.Enabled {
			h.renderSetup(w, r, code, "", "Choose one of the listed connections.")
			return
		}
		h.begin(w, r, conn, authRequest{ReturnTo: h.cfg.DefaultReturn, SetupCodeID: &codeID}, "")
	default:
		h.renderSetup(w, r, code, "", "")
	}
}

func (h *Handler) setupChoose(w http.ResponseWriter, r *http.Request, code string) {
	name, protocol, _ := strings.Cut(r.PostForm.Get("choice"), ":")
	p, ok := presets.Get(name)
	if !ok || !p.Supports(protocol) {
		page := web.SetupPage{Code: code, Step: "choose", Presets: choices(), PublicURL: h.cfg.PublicURL}
		h.cfg.Web.Render(w, r, http.StatusOK, web.PageSetup, "Set up single sign-on", page)
		return
	}
	h.renderConfigure(w, r, code, p, protocol, presets.Values{"jit": "on"}, "")
}

func (h *Handler) renderConfigure(w http.ResponseWriter, r *http.Request, code string, p presets.Preset, protocol string, v presets.Values, problem string) {
	page := web.SetupPage{Code: code, Step: "configure", Preset: p.Name, PresetLabel: p.Label, Protocol: protocol, PublicURL: h.cfg.PublicURL,
		Error: problem}
	for _, f := range p.FieldsFor(protocol) {
		if f.ConsoleOnly {
			continue
		}
		ff := web.FormField{Name: f.Name, Label: f.Label, Help: f.Help, Kind: f.Kind, Required: f.Required, Placeholder: f.Placeholder,
			Options: f.Options}
		if f.Kind != presets.KindSecret && !f.WriteOnly {
			ff.Value = v[f.Name]
		}
		page.Fields = append(page.Fields, ff)
	}
	h.cfg.Web.Render(w, r, http.StatusOK, web.PageSetup, "Set up single sign-on", page)
}

func (h *Handler) setupCreate(w http.ResponseWriter, r *http.Request, code string) {
	p, ok := presets.Get(r.PostForm.Get("preset"))
	protocol := r.PostForm.Get("protocol")
	if !ok || !p.Supports(protocol) {
		h.renderSetup(w, r, code, "", "Choose an identity provider first.")
		return
	}
	v := presets.FromForm(r.PostForm)
	in, err := p.Build(protocol, v)
	if err == nil {
		_, err = h.conns.Create(r.Context(), in, func(tx pgx.Tx, c connections.Connection) error {
			return audit.Append(r.Context(), tx, h.event(r, "sso.connection_created", "success", &c, nil, map[string]any{"via": "setup", "preset": p.Name}))
		})
		if err == nil {
			h.renderSetup(w, r, code, "Connection saved. Register the URLs below at your identity provider, then sign in.", "")
			return
		}
	}
	h.renderConfigure(w, r, code, p, protocol, v, setupProblem(err))
}

func setupProblem(err error) string {
	var ve *connections.ValidationError
	switch {
	case errors.As(err, &ve):
		return ve.Field + ": " + ve.Message
	case errors.Is(err, connections.ErrSlugTaken):
		return "That slug is already used by another connection."
	case errors.Is(err, connections.ErrDomainTaken):
		return "Another connection already routes one of those email domains."
	}
	return "The connection could not be saved: " + reason(err)
}
