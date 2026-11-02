package login

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// authTTL is how long a started sign-in may take to come back from the IdP.
const authTTL = 10 * time.Minute

// authRequest is one in-flight sign-in, stored server-side and bound to the browser by a cookie.
type authRequest struct {
	ID            string
	Nonce         string
	Verifier      string
	SAMLRequestID string
	ReturnTo      string
	MaxAge        int
	SetupCodeID   *string
}

func bindingCookie(state string, secure bool) string {
	sum := sha256.Sum256([]byte(state))
	name := "starlight_sso_" + hex.EncodeToString(sum[:8])
	if secure {
		return "__Host-" + name
	}
	return name
}

// save stores ar under the state's hash and sets the cookie that proves the callback reaches the same browser.
func (h *Handler) save(ctx context.Context, w http.ResponseWriter, c connections.Connection, state string, ar authRequest) error {
	binding := secret.RandomString(32)
	id := ids.New()
	var sealed []byte
	if ar.Verifier != "" {
		var err error
		if sealed, err = h.cfg.Keyring.Seal([]byte(ar.Verifier), "auth_requests.pkce:"+id); err != nil {
			return err
		}
	}
	var maxAge *int
	if ar.MaxAge > 0 {
		maxAge = &ar.MaxAge
	}
	_, err := h.cfg.Pool.Exec(ctx, `INSERT INTO auth_requests (id, state_hash, connection_id, binding_hash, nonce, pkce_verifier, saml_request_id,
		return_to, max_age_seconds, expires_at, setup_code_id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
		id, secret.Hash(state), c.ID, secret.Hash(binding), ar.Nonce, sealed, ar.SAMLRequestID, ar.ReturnTo, maxAge,
		h.cfg.Now().Add(authTTL), ar.SetupCodeID)
	if err != nil {
		return err
	}
	saml := c.Protocol == connections.SAML
	secure := h.secure || saml
	cookie := &http.Cookie{Name: bindingCookie(state, secure), Value: binding, Path: "/", HttpOnly: true, Secure: secure,
		MaxAge: int(authTTL.Seconds()), SameSite: http.SameSiteLaxMode}
	if saml {
		cookie.SameSite = http.SameSiteNoneMode
	}
	http.SetCookie(w, cookie)
	return nil
}

// consume returns the sign-in for state exactly once, only to the browser holding its binding cookie.
func (h *Handler) consume(ctx context.Context, w http.ResponseWriter, r *http.Request, c connections.Connection, state string) (authRequest, error) {
	if state == "" || len(state) > 128 {
		return authRequest{}, errStale
	}
	var ar authRequest
	var bindingHash, sealed []byte
	var maxAge *int
	var expires time.Time
	var used *time.Time
	err := h.cfg.Pool.QueryRow(ctx, `SELECT id, binding_hash, nonce, pkce_verifier, saml_request_id, return_to, max_age_seconds, setup_code_id,
		expires_at, used_at FROM auth_requests WHERE state_hash = $1 AND connection_id = $2`, secret.Hash(state), c.ID).
		Scan(&ar.ID, &bindingHash, &ar.Nonce, &sealed, &ar.SAMLRequestID, &ar.ReturnTo, &maxAge, &ar.SetupCodeID, &expires, &used)
	if db.NotFound(err) {
		return authRequest{}, errStale
	}
	if err != nil {
		return authRequest{}, err
	}
	if used != nil || !h.cfg.Now().Before(expires) {
		return authRequest{}, errStale
	}
	saml := c.Protocol == connections.SAML
	name := bindingCookie(state, h.secure || saml)
	cookie, err := r.Cookie(name)
	if err != nil || !secret.Equal(secret.Hash(cookie.Value), bindingHash) {
		return authRequest{}, errStale
	}
	tag, err := h.cfg.Pool.Exec(ctx, "UPDATE auth_requests SET used_at = now() WHERE id = $1 AND used_at IS NULL", ar.ID)
	if err != nil {
		return authRequest{}, err
	}
	if tag.RowsAffected() != 1 {
		return authRequest{}, errStale
	}
	http.SetCookie(w, &http.Cookie{Name: name, Value: "", Path: "/", HttpOnly: true, Secure: h.secure || saml, MaxAge: -1})
	if maxAge != nil {
		ar.MaxAge = *maxAge
	}
	if len(sealed) > 0 {
		v, err := h.cfg.Keyring.Open(sealed, "auth_requests.pkce:"+ar.ID)
		if err != nil {
			return authRequest{}, err
		}
		ar.Verifier = string(v)
	}
	return ar, nil
}
