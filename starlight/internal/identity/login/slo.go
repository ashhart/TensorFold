package login

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"html/template"
	"net/http"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// logoutTTL bounds how long Starlight waits for the IdP to answer a LogoutRequest it sent.
const logoutTTL = 10 * time.Minute

// autoSubmit posts an HTTP-POST binding form on load; the page's CSP allows this script and nothing else.
const autoSubmit = "document.forms[0].submit();"

var autoPostCSP = func() string {
	sum := sha256.Sum256([]byte(autoSubmit))
	return "default-src 'none'; script-src 'sha256-" + base64.StdEncoding.EncodeToString(sum[:]) + "'; base-uri 'none'; frame-ancestors 'none'"
}()

var autoPost = template.Must(template.New("autopost").Parse(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Signing out</title></head><body>
<form method="post" action="{{.URL}}">{{range $k, $v := .Form}}<input type="hidden" name="{{$k}}" value="{{index $v 0}}">{{end}}
<button type="submit">Continue signing out</button></form>
<script>` + autoSubmit + `</script></body></html>`))

var errNoLogoutRequest = errors.New("login: logout response answers no pending request; it is unknown, expired or already used")

// deliver sends the browser on: a redirect, or a self-submitting form for the HTTP-POST binding.
func deliver(w http.ResponseWriter, r *http.Request, out samlsp.Outbound) {
	if out.Form == nil {
		http.Redirect(w, r, out.URL, http.StatusSeeOther)
		return
	}
	var buf bytes.Buffer
	if err := autoPost.Execute(&buf, out); err != nil {
		http.Error(w, "logout form failed to render", http.StatusInternalServerError)
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/html; charset=utf-8")
	h.Set("Cache-Control", "no-store")
	h.Set("Content-Security-Policy", autoPostCSP)
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("Referrer-Policy", "same-origin")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(buf.Bytes())
}

func (h *Handler) sp(ctx context.Context, c connections.Connection) (*samlsp.SP, error) {
	secrets, err := h.conns.Secrets(ctx, c)
	if err != nil {
		return nil, err
	}
	return h.samls.SP(ctx, c, secrets)
}

// samlSignOut signs a LogoutRequest for the session when the connection opts in and the IdP lists a logout endpoint.
func (h *Handler) samlSignOut(r *http.Request, c connections.Connection, s session.Session, userID string) (*samlsp.Outbound, error) {
	if c.Config.SAML == nil || !c.Config.SAML.RPLogout || s.IDPSubject == "" {
		return nil, nil
	}
	sp, err := h.sp(r.Context(), c)
	if err != nil || !sp.HasSLO() {
		return nil, err
	}
	n := samlsp.NameID{Value: s.IDPSubject, Format: s.NameID.Format, NameQualifier: s.NameID.NameQualifier, SPNameQualifier: s.NameID.SPNameQualifier}
	out, id, err := sp.LogoutRequest(n, s.IDPSessionID)
	if err != nil {
		return nil, err
	}
	if _, err := h.cfg.Pool.Exec(r.Context(), `INSERT INTO saml_logout_requests (connection_id, request_id, user_id, expires_at)
		VALUES ($1, $2, $3, $4)`, c.ID, id, userID, h.cfg.Now().Add(logoutTTL)); err != nil {
		return nil, err
	}
	return &out, nil
}

// samlSLO is Starlight's logout endpoint: IdP-initiated LogoutRequests and answers to its own, over either binding.
func (h *Handler) samlSLO(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if !h.samlLimiter.allow(r) {
		h.fail(w, r, http.StatusTooManyRequests, "Too many attempts", "Too many sign-out requests from your network. Wait a minute and try again.")
		return
	}
	conn, err := h.conns.GetBySlug(r.Context(), r.PathValue("slug"))
	if err != nil || conn.Protocol != connections.SAML {
		h.fail(w, r, http.StatusNotFound, "Unknown sign-in method", "That sign-in method does not exist.")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
	m, err := samlsp.ReadMessage(r)
	if err != nil {
		h.logoutRefused(w, r, &conn, "auth.idp_logout", "binding", err)
		return
	}
	sp, err := h.sp(r.Context(), conn)
	if err != nil {
		h.logoutRefused(w, r, &conn, "auth.idp_logout", "metadata", err)
		return
	}
	if m.Request {
		h.samlIDPLogout(w, r, conn, sp, m)
		return
	}
	h.samlLogoutResponse(w, r, conn, sp, m)
}

// samlIDPLogout ends every session the IdP names, then answers with a signed LogoutResponse.
func (h *Handler) samlIDPLogout(w http.ResponseWriter, r *http.Request, conn connections.Connection, sp *samlsp.SP, m samlsp.Message) {
	req, err := sp.VerifyLogoutRequest(m)
	if err != nil {
		h.logoutRefused(w, r, &conn, "auth.idp_logout", "verify", err)
		return
	}
	if err := h.once(r, conn.ID, "saml_logout", req.ID, req.Expires); err != nil {
		h.logoutRefused(w, r, &conn, "auth.idp_logout", "replay", err)
		return
	}
	n, err := session.RevokeSAML(r.Context(), h.cfg.Pool, conn.ID, req.NameID, req.SessionIndexes)
	if err != nil {
		h.fail(w, r, http.StatusInternalServerError, "Sign-out failed", "Starlight could not end the sessions. Try again.")
		return
	}
	h.record(r, "auth.idp_logout", "success", &conn, nil, map[string]any{"sessions": n, "by_session_index": len(req.SessionIndexes) > 0,
		"binding": m.Binding()})
	out, err := sp.LogoutResponse(req.ID, m.RelayState)
	if err != nil {
		httpxLog(r, "saml logout response not sent", err)
		h.cfg.Web.Message(w, r, http.StatusOK, "success", "Signed out", "Your Starlight sessions from this identity provider have ended.",
			web.Link{Href: "/auth/login", Label: "Sign in again"})
		return
	}
	deliver(w, r, out)
}

// samlLogoutResponse finishes a sign-out Starlight started; its session ended before the browser went to the IdP.
func (h *Handler) samlLogoutResponse(w http.ResponseWriter, r *http.Request, conn connections.Connection, sp *samlsp.SP, m samlsp.Message) {
	resp, err := sp.VerifyLogoutResponse(m)
	if err != nil {
		h.logoutRefused(w, r, &conn, "auth.logout_response", "verify", err)
		return
	}
	var userID *string
	err = h.cfg.Pool.QueryRow(r.Context(), `DELETE FROM saml_logout_requests WHERE connection_id = $1 AND request_id = $2 AND expires_at > $3
		RETURNING user_id`, conn.ID, resp.InResponseTo, h.cfg.Now()).Scan(&userID)
	if db.NotFound(err) {
		err = errNoLogoutRequest
	}
	if err != nil {
		h.logoutRefused(w, r, &conn, "auth.logout_response", "request", err)
		return
	}
	var user *directory.User
	if userID != nil {
		if u, err := directory.GetUser(r.Context(), h.cfg.Pool, *userID); err == nil {
			user = &u
		}
	}
	if !resp.Success() {
		h.record(r, "auth.logout_response", "failure", &conn, user, map[string]any{"stage": "status", "status": resp.Status})
		h.cfg.Web.Message(w, r, http.StatusOK, "error", "Sign-out incomplete",
			"You are signed out of Starlight, but your identity provider could not end its own session. Close your browser to finish signing out.",
			web.Link{Href: "/auth/login", Label: "Back to sign-in"})
		return
	}
	h.record(r, "auth.logout_response", "success", &conn, user, nil)
	http.Redirect(w, r, "/auth/login?signed_out=1", http.StatusSeeOther)
}

// logoutRefused audits a logout message Starlight won't act on and tells the browser without saying which check failed.
func (h *Handler) logoutRefused(w http.ResponseWriter, r *http.Request, conn *connections.Connection, action, stage string, err error) {
	httpxLog(r, "saml logout refused at "+stage, err)
	h.record(r, action, "failure", conn, nil, map[string]any{"stage": stage, "reason": reason(err)})
	if action == "auth.logout_response" {
		h.fail(w, r, http.StatusBadRequest, "Sign-out not confirmed",
			"You are signed out of Starlight, but Starlight could not verify your identity provider's reply. Close your browser to finish signing out.")
		return
	}
	h.fail(w, r, http.StatusBadRequest, "Sign-out request refused", "Starlight could not verify the sign-out request from your identity provider.")
}
