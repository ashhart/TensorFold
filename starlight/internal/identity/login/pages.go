package login

import (
	"net/http"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

func (h *Handler) loginData(r *http.Request, returnTo string) (web.LoginPage, error) {
	all, err := h.conns.List(r.Context())
	if err != nil {
		return web.LoginPage{}, err
	}
	page := web.LoginPage{ReturnTo: returnTo}
	for _, c := range all {
		if !c.Enabled {
			continue
		}
		page.Options = append(page.Options, web.LoginOption{Slug: c.Slug, Name: c.Name, Preset: c.Preset, Protocol: c.Protocol})
		if len(c.Domains) > 0 {
			page.ShowEmail = true
		}
	}
	return page, nil
}

// loginPage lists the enabled connections, with email routing when any connection claims domains.
func (h *Handler) loginPage(w http.ResponseWriter, r *http.Request) {
	page, err := h.loginData(r, session.SafeReturn(r.URL.Query().Get("return_to"), h.cfg.DefaultReturn))
	if err != nil {
		h.fail(w, r, http.StatusServiceUnavailable, "Sign-in unavailable", "Starlight cannot load its sign-in methods right now.")
		return
	}
	if r.URL.Query().Get("signed_out") == "1" {
		page.Notice = "You have signed out."
	}
	h.cfg.Web.Render(w, r, http.StatusOK, web.PageLogin, "Sign in", page)
}

// loginEmail starts SSO at the connection that owns the email's domain, keeping the address out of Starlight's URLs.
func (h *Handler) loginEmail(w http.ResponseWriter, r *http.Request) {
	if !session.SameOrigin(r) || !h.limiter.allow(r) {
		h.fail(w, r, http.StatusForbidden, "Request refused", "Start again from the sign-in page.")
		return
	}
	email := strings.TrimSpace(r.PostFormValue("email"))
	returnTo := session.SafeReturn(r.PostFormValue("return_to"), h.cfg.DefaultReturn)
	conn, err := h.conns.ForEmail(r.Context(), email)
	if err != nil {
		page, lerr := h.loginData(r, returnTo)
		if lerr != nil {
			h.fail(w, r, http.StatusServiceUnavailable, "Sign-in unavailable", "Starlight cannot load its sign-in methods right now.")
			return
		}
		page.Email, page.Error = email, "No sign-in method is set up for that email domain. Choose one below or ask your administrator."
		h.cfg.Web.Render(w, r, http.StatusOK, web.PageLogin, "Sign in", page)
		return
	}
	h.begin(w, r, conn, authRequest{ReturnTo: returnTo}, loginHint(email))
}

// signedOut is the page every sign-out ends on, whether or not the IdP took part.
const signedOut = "/auth/login?signed_out=1"

// logout ends the Starlight session and, when the connection asks for it, the IdP session too.
func (h *Handler) logout(w http.ResponseWriter, r *http.Request) {
	if !session.SameOrigin(r) {
		h.fail(w, r, http.StatusForbidden, "Request refused", "Sign out from Starlight's own pages.")
		return
	}
	s, u, err := h.cfg.Sessions.Load(r.Context(), r)
	if err == nil && !s.ValidCSRF(r.PostFormValue("_csrf")) {
		h.fail(w, r, http.StatusForbidden, "Request refused", "Your sign-out form expired. Reload the page and try again.")
		return
	}
	if derr := h.cfg.Sessions.Destroy(r.Context(), w, r); derr != nil {
		h.fail(w, r, http.StatusInternalServerError, "Sign-out failed", "Starlight could not end your session. Try again.")
		return
	}
	out := samlsp.Outbound{URL: signedOut}
	if err == nil {
		var conn *connections.Connection
		var details map[string]any
		if s.ConnectionID != nil {
			if c, cerr := h.conns.Get(r.Context(), *s.ConnectionID); cerr == nil {
				conn = &c
				var next *samlsp.Outbound
				if next, details = h.idpLogout(r, c, s, u.ID); next != nil {
					out = *next
				}
			}
		}
		h.record(r, "auth.logout", "success", conn, &u, details)
	}
	deliver(w, r, out)
}

// idpLogout returns where the browser goes to end the IdP session too, or nil to stop at Starlight, with audit details.
func (h *Handler) idpLogout(r *http.Request, c connections.Connection, s session.Session, userID string) (*samlsp.Outbound, map[string]any) {
	if c.Protocol == connections.SAML {
		out, err := h.samlSignOut(r, c, s, userID)
		switch {
		case err != nil:
			httpxLog(r, "saml logout request not sent; signed out of Starlight only", err)
			return nil, map[string]any{"idp_logout": "failed", "reason": reason(err)}
		case out != nil:
			return out, map[string]any{"idp_logout": "saml"}
		}
		return nil, nil
	}
	if c.Config.OIDC == nil || !c.Config.OIDC.RPLogout {
		return nil, nil
	}
	secrets, err := h.conns.Secrets(r.Context(), c)
	if err != nil {
		return nil, nil
	}
	client, err := h.rp.Client(r.Context(), c, secrets, h.redirectURI(c.Slug))
	if err != nil {
		return nil, nil
	}
	if u := client.EndSessionURL(h.cfg.PublicURL + signedOut); u != "" {
		return &samlsp.Outbound{URL: u}, map[string]any{"idp_logout": "oidc"}
	}
	return nil, nil
}
