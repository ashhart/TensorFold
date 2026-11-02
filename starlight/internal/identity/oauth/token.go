package oauth

import (
	"errors"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

const deviceGrantType = "urn:ietf:params:oauth:grant-type:device_code"

func (s *Server) token(w http.ResponseWriter, r *http.Request) {
	if ok, _ := s.tokenRate.Take(r.Context(), s.clientKey(r)); !ok {
		writeOAuthError(w, errSlowDown)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
	if err := r.ParseForm(); err != nil {
		writeOAuthError(w, invalidRequest("form body required"))
		return
	}
	client, ok := LookupClient(r.PostForm.Get("client_id"))
	if !ok {
		writeOAuthError(w, errInvalidClient)
		return
	}
	var g Grant
	var e *oauthError
	switch r.PostForm.Get("grant_type") {
	case "authorization_code":
		g, e = s.exchangeCode(r, client)
	case "refresh_token":
		g, e = s.refresh(r, client)
	case deviceGrantType:
		g, e = s.pollDevice(r, client)
	default:
		e = &oauthError{http.StatusBadRequest, "unsupported_grant_type", "grant_type is not supported"}
	}
	if e != nil {
		writeOAuthError(w, *e)
		return
	}
	w.Header().Set("Pragma", "no-cache")
	httpx.JSON(w, http.StatusOK, discovery.TokenResponse{AccessToken: g.Access, TokenType: "Bearer", ExpiresIn: int(g.Expires.Seconds()),
		RefreshToken: g.Refresh, Scope: g.Scope})
}

func (s *Server) exchangeCode(r *http.Request, client Client) (Grant, *oauthError) {
	code, verifier, redirect := r.PostForm.Get("code"), r.PostForm.Get("code_verifier"), r.PostForm.Get("redirect_uri")
	if code == "" || verifier == "" {
		e := invalidRequest("code and code_verifier are required")
		return Grant{}, &e
	}
	var g Grant
	burned := false
	err := db.Tx(r.Context(), s.Pool, func(tx pgx.Tx) error {
		var userID, codeClient, codeRedirect, challenge, scope, device, status string
		err := tx.QueryRow(r.Context(), `UPDATE oauth_codes c SET used_at = now() FROM users u
			WHERE c.code_hash = $1 AND c.used_at IS NULL AND c.expires_at > now() AND u.id = c.user_id
			RETURNING c.user_id, c.client_id, c.redirect_uri, c.code_challenge, c.scope, c.device_name, u.status`, secret.Hash(code)).
			Scan(&userID, &codeClient, &codeRedirect, &challenge, &scope, &device, &status)
		if db.NotFound(err) {
			return ErrInvalidGrant
		}
		if err != nil {
			return err
		}
		if codeClient != client.ID || codeRedirect != redirect || status != "active" || !pkceMatches(verifier, challenge) {
			burned = true
			return nil
		}
		g, err = s.Issuer.Issue(r.Context(), tx, userID, client.ID, scope, device)
		return err
	})
	if burned || errors.Is(err, ErrInvalidGrant) {
		return Grant{}, &errInvalidGrant
	}
	if err != nil {
		return Grant{}, &oauthError{http.StatusInternalServerError, "server_error", "try again"}
	}
	return g, nil
}

func (s *Server) refresh(r *http.Request, client Client) (Grant, *oauthError) {
	g, userID, err := s.Issuer.Rotate(r.Context(), r.PostForm.Get("refresh_token"), client.ID)
	if errors.Is(err, ErrInvalidGrant) {
		if userID != "" {
			s.record(r, "oauth.refresh_reuse", "denied", map[string]any{"client": client.ID, "user_id": userID})
		}
		return Grant{}, &errInvalidGrant
	}
	if err != nil {
		return Grant{}, &oauthError{http.StatusInternalServerError, "server_error", "try again"}
	}
	return g, nil
}

func (s *Server) pollDevice(r *http.Request, client Client) (Grant, *oauthError) {
	if !client.DeviceGrant {
		return Grant{}, &oauthError{http.StatusBadRequest, "unauthorized_client", "client may not use the device grant"}
	}
	code := r.PostForm.Get("device_code")
	if k, ok := secret.Parse(code); !ok || k != secret.Device {
		return Grant{}, &errInvalidGrant
	}
	var g Grant
	var oe *oauthError
	err := db.Tx(r.Context(), s.Pool, func(tx pgx.Tx) error {
		var id, grantClient, scope, device, status string
		var userID *string
		var interval int
		var last *time.Time
		var expires time.Time
		err := tx.QueryRow(r.Context(), `SELECT id, client_id, scope, device_name, status, user_id, interval_seconds, last_polled_at, expires_at
			FROM device_grants WHERE device_code_hash = $1 FOR UPDATE`, secret.Hash(code)).
			Scan(&id, &grantClient, &scope, &device, &status, &userID, &interval, &last, &expires)
		if db.NotFound(err) || err == nil && grantClient != client.ID {
			oe = &errInvalidGrant
			return nil
		}
		if err != nil {
			return err
		}
		if time.Now().After(expires) {
			oe = &oauthError{http.StatusBadRequest, "expired_token", "the device code expired"}
			return nil
		}
		if last != nil && time.Since(*last) < time.Duration(interval)*time.Second {
			oe = &oauthError{http.StatusBadRequest, "slow_down", "poll less often"}
			_, err := tx.Exec(r.Context(), "UPDATE device_grants SET interval_seconds = least(interval_seconds + 5, 60), last_polled_at = now() WHERE id = $1", id)
			return err
		}
		if _, err := tx.Exec(r.Context(), "UPDATE device_grants SET last_polled_at = now() WHERE id = $1", id); err != nil {
			return err
		}
		switch status {
		case "pending":
			oe = &oauthError{http.StatusBadRequest, "authorization_pending", "waiting for the user"}
			return nil
		case "denied":
			oe = &oauthError{http.StatusBadRequest, "access_denied", "the user denied the request"}
			return nil
		case "consumed":
			oe = &errInvalidGrant
			return nil
		}
		var userStatus string
		if err := tx.QueryRow(r.Context(), "SELECT status FROM users WHERE id = $1", *userID).Scan(&userStatus); err != nil || userStatus != "active" {
			oe = &errInvalidGrant
			return nil
		}
		if _, err := tx.Exec(r.Context(), "UPDATE device_grants SET status = 'consumed' WHERE id = $1", id); err != nil {
			return err
		}
		g, err = s.Issuer.Issue(r.Context(), tx, *userID, client.ID, scope, device)
		return err
	})
	if err != nil {
		return Grant{}, &oauthError{http.StatusInternalServerError, "server_error", "try again"}
	}
	return g, oe
}

func (s *Server) revoke(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 16<<10)
	if err := r.ParseForm(); err != nil {
		writeOAuthError(w, invalidRequest("form body required"))
		return
	}
	if _, ok := LookupClient(r.PostForm.Get("client_id")); !ok {
		writeOAuthError(w, errInvalidClient)
		return
	}
	if err := s.Issuer.Revoke(r.Context(), r.PostForm.Get("token"), r.PostForm.Get("client_id")); err != nil {
		writeOAuthError(w, oauthError{http.StatusInternalServerError, "server_error", "try again"})
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (s *Server) userinfo(w http.ResponseWriter, r *http.Request) {
	tok := httpx.BearerToken(r)
	userID, err := s.AccessFor(r.Context(), tok)
	if err != nil {
		httpx.Write(w, httpx.Plain, httpx.ErrUnauthenticated)
		return
	}
	me, err := s.Me(r.Context(), userID)
	if err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	httpx.JSON(w, http.StatusOK, me)
}
