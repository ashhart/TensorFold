package login

import (
	"errors"
	"net/http"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
)

// errReplay means an assertion or logout token was already used.
var errReplay = errors.New("login: replayed assertion")

func (h *Handler) beginSAML(w http.ResponseWriter, r *http.Request, conn connections.Connection, ar authRequest) (string, error) {
	secrets, err := h.conns.Secrets(r.Context(), conn)
	if err != nil {
		return "", err
	}
	sp, err := h.samls.SP(r.Context(), conn, secrets)
	if err != nil {
		return "", err
	}
	state := secret.RandomString(32)
	target, requestID, err := sp.AuthnRedirect(state)
	if err != nil {
		return "", err
	}
	ar.SAMLRequestID = requestID
	if err := h.save(r.Context(), w, conn, state, ar); err != nil {
		return "", err
	}
	return target, nil
}

// samlMetadata publishes Starlight's SP metadata for the IdP administrator.
func (h *Handler) samlMetadata(w http.ResponseWriter, r *http.Request) {
	conn, err := h.conns.GetBySlug(r.Context(), r.PathValue("slug"))
	if err != nil || conn.Protocol != connections.SAML {
		http.NotFound(w, r)
		return
	}
	secrets, err := h.conns.Secrets(r.Context(), conn)
	if err != nil {
		http.Error(w, "connection secrets unavailable", http.StatusInternalServerError)
		return
	}
	sp, err := h.samls.SP(r.Context(), conn, secrets)
	if err != nil {
		httpxLog(r, "saml metadata", err)
		http.Error(w, "IdP metadata unavailable; check the connection", http.StatusBadGateway)
		return
	}
	body, err := sp.Metadata()
	if err != nil {
		http.Error(w, "metadata failed", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/samlmetadata+xml")
	w.Header().Set("Cache-Control", "public, max-age=3600")
	_, _ = w.Write(body)
}

// samlACS receives the IdP's POST and finishes the sign-in.
func (h *Handler) samlACS(w http.ResponseWriter, r *http.Request) {
	if !h.samlLimiter.allow(r) {
		h.fail(w, r, http.StatusTooManyRequests, "Too many attempts", "Too many sign-in attempts from your network. Wait a minute and try again.")
		return
	}
	conn, err := h.conns.GetBySlug(r.Context(), r.PathValue("slug"))
	if err != nil || !conn.Enabled || conn.Protocol != connections.SAML {
		h.fail(w, r, http.StatusNotFound, "Unknown sign-in method", "That sign-in method does not exist or is turned off.")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
	if err := r.ParseForm(); err != nil {
		h.refuse(w, r, &conn, "form", err)
		return
	}
	if r.PostForm.Get("SAMLart") != "" || r.PostForm.Get("SAMLResponse") == "" {
		h.refuse(w, r, &conn, "binding", errors.New("only the HTTP-POST binding with a SAMLResponse is accepted"))
		return
	}
	relay := r.PostForm.Get("RelayState")
	ar, possible, err := h.samlRequest(w, r, conn, relay)
	if err != nil {
		h.refuse(w, r, &conn, "state", err)
		return
	}
	secrets, err := h.conns.Secrets(r.Context(), conn)
	if err != nil {
		h.refuse(w, r, &conn, "secrets", err)
		return
	}
	sp, err := h.samls.SP(r.Context(), conn, secrets)
	if err != nil {
		h.refuse(w, r, &conn, "metadata", err)
		return
	}
	a, err := sp.Parse(r.PostForm.Get("SAMLResponse"), possible)
	if err != nil {
		h.refuse(w, r, &conn, "assertion", err)
		return
	}
	if err := h.once(r, conn.ID, "saml_assertion", a.ID, a.NotOnOrAfter); err != nil {
		h.refuse(w, r, &conn, "replay", err)
		return
	}
	id, err := mapping.FromSAML(conn, mapping.SAMLInput{NameID: a.NameID, NameIDFormat: a.NameIDFormat, Attributes: a.Attributes,
		SessionIndex: a.SessionIndex, AuthInstant: a.AuthInstant})
	if err != nil {
		h.refuse(w, r, &conn, "claims", err)
		return
	}
	h.finish(w, r, conn, secrets, ar, id, session.NameID{Format: a.NameIDFormat, NameQualifier: a.NameQualifier, SPNameQualifier: a.SPNameQualifier})
}

// samlRequest matches RelayState to a stored sign-in; unmatched posts count as IdP-initiated only when allowed.
func (h *Handler) samlRequest(w http.ResponseWriter, r *http.Request, conn connections.Connection, relay string) (authRequest, []string, error) {
	if relay != "" {
		ar, err := h.consume(r.Context(), w, r, conn, relay)
		if err == nil {
			return ar, []string{ar.SAMLRequestID}, nil
		}
		if !errors.Is(err, errStale) || !conn.Config.SAML.AllowIDPInitiated {
			return authRequest{}, nil, err
		}
	}
	if !conn.Config.SAML.AllowIDPInitiated {
		return authRequest{}, nil, errStale
	}
	return authRequest{ReturnTo: session.SafeReturn(relay, h.cfg.DefaultReturn)}, nil, nil
}

// once records a single-use value until it expires and fails if it was seen before.
func (h *Handler) once(r *http.Request, connectionID, kind, value string, expires time.Time) error {
	floor := h.cfg.Now().Add(10 * time.Minute)
	expires = expires.Add(5 * time.Minute)
	if expires.Before(floor) {
		expires = floor
	}
	tag, err := h.cfg.Pool.Exec(r.Context(), `INSERT INTO sso_replay (connection_id, kind, value, expires_at) VALUES ($1, $2, $3, $4)
		ON CONFLICT (connection_id, kind, value) DO NOTHING`, connectionID, kind, value, expires)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return errReplay
	}
	return nil
}
