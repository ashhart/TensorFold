package oauth

import (
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
)

type authzRequest struct {
	client    Client
	redirect  string
	state     string
	challenge string
	scope     string
	device    string
}

func parseAuthorize(v url.Values) (authzRequest, string) {
	c, ok := LookupClient(v.Get("client_id"))
	if !ok {
		return authzRequest{}, "unknown client"
	}
	if !c.ValidRedirect(v.Get("redirect_uri")) {
		return authzRequest{}, "redirect_uri must be a loopback IP address with a port"
	}
	if v.Get("response_type") != "code" {
		return authzRequest{}, "response_type must be code"
	}
	if v.Get("code_challenge_method") != "S256" || len(v.Get("code_challenge")) != 43 {
		return authzRequest{}, "PKCE with S256 is required"
	}
	if s := v.Get("state"); len(s) < 8 || len(s) > 512 {
		return authzRequest{}, "state is required"
	}
	scope, ok := c.NarrowScope(v.Get("scope"))
	if !ok {
		return authzRequest{}, "scope is not available to this client"
	}
	return authzRequest{client: c, redirect: v.Get("redirect_uri"), state: v.Get("state"), challenge: v.Get("code_challenge"),
		scope: scope, device: DeviceName(v.Get("device_name"))}, ""
}

func (s *Server) authorizeForm(w http.ResponseWriter, r *http.Request) {
	ar, problem := parseAuthorize(r.URL.Query())
	if problem != "" {
		s.Pages.Message(w, r, MessagePage{Status: http.StatusBadRequest, Title: "Sign-in request refused", Body: problem})
		return
	}
	sess, u, _ := session.From(r.Context())
	hidden := map[string]string{}
	for _, k := range []string{"client_id", "redirect_uri", "response_type", "code_challenge", "code_challenge_method", "state", "scope", "device_name"} {
		hidden[k] = r.URL.Query().Get(k)
	}
	s.Pages.Approve(w, r, ApprovePage{OrgName: s.org(r.Context()), ClientName: ar.client.Name, DeviceName: ar.device, UserEmail: u.Email,
		Scopes: strings.Fields(ar.scope), Action: "/oauth/authorize", CSRF: sess.CSRF(), Hidden: hidden})
}

func (s *Server) authorizeDecide(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	ar, problem := parseAuthorize(r.PostForm)
	if problem != "" {
		s.Pages.Message(w, r, MessagePage{Status: http.StatusBadRequest, Title: "Sign-in request refused", Body: problem})
		return
	}
	sess, u, _ := session.From(r.Context())
	target, _ := url.Parse(ar.redirect)
	q := target.Query()
	q.Set("state", ar.state)
	if r.PostFormValue("decision") != "allow" {
		q.Set("error", "access_denied")
		target.RawQuery = q.Encode()
		s.record(r, "oauth.authorize", "denied", map[string]any{"client": ar.client.ID, "device": ar.device})
		http.Redirect(w, r, target.String(), http.StatusSeeOther)
		return
	}
	code := secret.RandomString(32)
	_, err := s.Pool.Exec(r.Context(), `INSERT INTO oauth_codes (code_hash, client_id, user_id, session_id, redirect_uri, code_challenge, scope,
		device_name, expires_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`, secret.Hash(code), ar.client.ID, u.ID, sess.ID, ar.redirect,
		ar.challenge, ar.scope, ar.device, time.Now().Add(2*time.Minute))
	if err != nil {
		s.Pages.Message(w, r, MessagePage{Status: http.StatusInternalServerError, Title: "Sign-in failed", Body: "Try again."})
		return
	}
	q.Set("code", code)
	target.RawQuery = q.Encode()
	s.record(r, "oauth.authorize", "success", map[string]any{"client": ar.client.ID, "device": ar.device, "scope": ar.scope})
	http.Redirect(w, r, target.String(), http.StatusSeeOther)
}

func pkceMatches(verifier, challenge string) bool {
	if len(verifier) < 43 || len(verifier) > 128 {
		return false
	}
	for _, c := range verifier {
		if !(c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || strings.ContainsRune("-._~", c)) {
			return false
		}
	}
	sum := sha256.Sum256([]byte(verifier))
	return secret.Equal([]byte(base64.RawURLEncoding.EncodeToString(sum[:])), []byte(challenge))
}
