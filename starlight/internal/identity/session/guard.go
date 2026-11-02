package session

import (
	"context"
	"net/http"
	"net/url"
	"path"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

type ctxKey struct{}

type loaded struct {
	s Session
	u directory.User
}

// From returns the session and user attached by Require.
func From(ctx context.Context) (Session, directory.User, bool) {
	l, ok := ctx.Value(ctxKey{}).(loaded)
	return l.s, l.u, ok
}

// Require redirects anonymous browsers to sign-in and enforces CSRF and same-origin on unsafe methods.
func (m *Manager) Require(loginPath string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s, u, err := m.Load(r.Context(), r)
		if err != nil {
			if r.Method != http.MethodGet && r.Method != http.MethodHead {
				http.Error(w, "session expired; sign in again", http.StatusUnauthorized)
				return
			}
			http.Redirect(w, r, loginPath+"?return_to="+url.QueryEscape(r.URL.RequestURI()), http.StatusSeeOther)
			return
		}
		if !safeMethod(r.Method) {
			if !SameOrigin(r) {
				http.Error(w, "cross-origin request refused", http.StatusForbidden)
				return
			}
			tok := r.Header.Get("X-CSRF-Token")
			if tok == "" {
				tok = r.PostFormValue("_csrf")
			}
			if !s.ValidCSRF(tok) {
				http.Error(w, "invalid or missing CSRF token", http.StatusForbidden)
				return
			}
		}
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxKey{}, loaded{s, u})))
	})
}

func safeMethod(m string) bool {
	return m == http.MethodGet || m == http.MethodHead || m == http.MethodOptions
}

// SameOrigin trusts a browser's same-origin Sec-Fetch-Site and otherwise needs an Origin naming this host.
func SameOrigin(r *http.Request) bool {
	switch r.Header.Get("Sec-Fetch-Site") {
	case "same-origin":
		return true
	case "", "none":
	default:
		return false
	}
	origin := r.Header.Get("Origin")
	if origin == "" || origin == "null" {
		return origin == ""
	}
	u, err := url.Parse(origin)
	return err == nil && strings.EqualFold(u.Host, r.Host)
}

// SafeReturn keeps post-login redirects on this site; backslashes and paths that clean to "//" are refused.
func SafeReturn(target, fallback string) string {
	if target == "" || !strings.HasPrefix(target, "/") || strings.ContainsAny(target, "\\\r\n") {
		return fallback
	}
	u, err := url.Parse(target)
	if err != nil || u.Scheme != "" || u.Host != "" || u.Opaque != "" || strings.Contains(u.Path, "\\") {
		return fallback
	}
	if clean := path.Clean(u.Path); strings.HasPrefix(clean, "//") || !strings.HasPrefix(clean, "/") {
		return fallback
	}
	return target
}
