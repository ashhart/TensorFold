package login

import (
	"net/http"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
)

// backchannel is OIDC Back-Channel Logout: the IdP posts a signed logout token and Starlight ends matching sessions.
func (h *Handler) backchannel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	bad := func() { httpx.JSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_request"}) }
	conn, err := h.conns.GetBySlug(r.Context(), r.PathValue("slug"))
	if err != nil || !conn.Enabled || conn.Protocol != connections.OIDC {
		bad()
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
	if err := r.ParseForm(); err != nil {
		bad()
		return
	}
	lt, err := h.rp.VerifyLogout(r.Context(), conn, r.PostForm.Get("logout_token"))
	if err != nil {
		httpxLog(r, "back-channel logout refused", err)
		h.record(r, "auth.idp_logout", "failure", &conn, nil, map[string]any{"reason": reason(err)})
		bad()
		return
	}
	expires := lt.Expiry
	if expires.IsZero() {
		expires = h.cfg.Now().Add(10 * time.Minute)
	}
	if err := h.once(r, conn.ID, "logout_jti", lt.TokenID, expires); err != nil {
		h.record(r, "auth.idp_logout", "failure", &conn, nil, map[string]any{"reason": "replayed jti"})
		bad()
		return
	}
	n, err := session.RevokeIDP(r.Context(), h.cfg.Pool, conn.ID, lt.SessionID, lt.Subject)
	if err != nil {
		httpx.JSON(w, http.StatusInternalServerError, map[string]string{"error": "server_error"})
		return
	}
	h.record(r, "auth.idp_logout", "success", &conn, nil, map[string]any{"sessions": n, "by_sid": lt.SessionID != ""})
	w.WriteHeader(http.StatusOK)
}
