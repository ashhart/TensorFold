package login

import (
	"net/http"
	"strconv"
	"strings"

	"golang.org/x/oauth2"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
)

// start sends the browser to the connection's identity provider.
func (h *Handler) start(w http.ResponseWriter, r *http.Request) {
	if !h.limiter.allow(r) {
		h.fail(w, r, http.StatusTooManyRequests, "Too many attempts", "Too many sign-in attempts from your network. Wait a minute and try again.")
		return
	}
	conn, err := h.conns.GetBySlug(r.Context(), r.PathValue("slug"))
	if err != nil || !conn.Enabled {
		h.fail(w, r, http.StatusNotFound, "Unknown sign-in method", "That sign-in method does not exist or is turned off.")
		return
	}
	q := r.URL.Query()
	ar := authRequest{ReturnTo: session.SafeReturn(q.Get("return_to"), h.cfg.DefaultReturn)}
	if n, err := strconv.Atoi(q.Get("max_age")); err == nil && n > 0 && n <= 86400 {
		ar.MaxAge = n
	}
	h.begin(w, r, conn, ar, loginHint(q.Get("login_hint")))
}

// begin starts SSO for an already validated request; setup and the email form also use it.
func (h *Handler) begin(w http.ResponseWriter, r *http.Request, conn connections.Connection, ar authRequest, hint string) {
	var target string
	var err error
	if conn.Protocol == connections.SAML {
		target, err = h.beginSAML(w, r, conn, ar)
	} else {
		target, err = h.beginOIDC(w, r, conn, ar, hint)
	}
	if err != nil {
		httpxLog(r, "sign-in start failed", err)
		h.record(r, "auth.login", "failure", &conn, nil, map[string]any{"stage": "start", "reason": reason(err)})
		h.fail(w, r, http.StatusBadGateway, "Identity provider unavailable",
			"Starlight could not reach your identity provider. Try again shortly; if it persists, ask your administrator.")
		return
	}
	http.Redirect(w, r, target, http.StatusSeeOther)
}

func (h *Handler) beginOIDC(w http.ResponseWriter, r *http.Request, conn connections.Connection, ar authRequest, hint string) (string, error) {
	secrets, err := h.conns.Secrets(r.Context(), conn)
	if err != nil {
		return "", err
	}
	client, err := h.rp.Client(r.Context(), conn, secrets, h.redirectURI(conn.Slug))
	if err != nil {
		return "", err
	}
	state := secret.RandomString(32)
	ar.Nonce, ar.Verifier = secret.RandomString(24), oauth2.GenerateVerifier()
	if err := h.save(r.Context(), w, conn, state, ar); err != nil {
		return "", err
	}
	return client.AuthURL(state, ar.Nonce, ar.Verifier, oidc.AuthOptions{LoginHint: hint, MaxAge: ar.MaxAge}), nil
}

// oidcCallback finishes an authorization-code sign-in.
func (h *Handler) oidcCallback(w http.ResponseWriter, r *http.Request) {
	conn, err := h.conns.GetBySlug(r.Context(), r.PathValue("slug"))
	if err != nil || !conn.Enabled || conn.Protocol != connections.OIDC {
		h.fail(w, r, http.StatusNotFound, "Unknown sign-in method", "That sign-in method does not exist or is turned off.")
		return
	}
	q := r.URL.Query()
	if code := q.Get("error"); code != "" {
		if !oauthErrorCode.MatchString(code) {
			code = "error"
		}
		h.record(r, "auth.login", "failure", &conn, nil, map[string]any{"stage": "idp", "idp_error": code})
		h.fail(w, r, http.StatusUnauthorized, "Sign-in cancelled", "Your identity provider did not complete the sign-in ("+code+").")
		return
	}
	ar, err := h.consume(r.Context(), w, r, conn, q.Get("state"))
	if err != nil {
		h.refuse(w, r, &conn, "state", err)
		return
	}
	secrets, err := h.conns.Secrets(r.Context(), conn)
	if err != nil {
		h.refuse(w, r, &conn, "secrets", err)
		return
	}
	client, err := h.rp.Client(r.Context(), conn, secrets, h.redirectURI(conn.Slug))
	if err != nil {
		h.refuse(w, r, &conn, "discovery", err)
		return
	}
	res, err := client.Exchange(r.Context(), q.Get("code"), ar.Verifier, ar.Nonce, ar.MaxAge)
	if err != nil {
		h.refuse(w, r, &conn, "token", err)
		return
	}
	id, err := mapping.FromClaims(conn, res.Subject, res.Claims)
	if err != nil {
		h.refuse(w, r, &conn, "claims", err)
		return
	}
	h.finish(w, r, conn, secrets, ar, id, session.NameID{})
}

// refuse records a failed sign-in and shows the user what they can do about it.
func (h *Handler) refuse(w http.ResponseWriter, r *http.Request, conn *connections.Connection, stage string, err error) {
	httpxLog(r, "sign-in refused at "+stage, err)
	h.record(r, "auth.login", "failure", conn, nil, map[string]any{"stage": stage, "reason": reason(err)})
	e := explain(err)
	h.fail(w, r, e.status, e.heading, e.body)
}

func loginHint(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 320 || strings.ContainsAny(s, "\r\n\t") {
		return ""
	}
	return s
}
