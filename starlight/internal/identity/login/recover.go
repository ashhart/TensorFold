package login

import (
	"net/http"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// recoverPage shows a confirmation, so link scanners that fetch the URL cannot spend the code.
func (h *Handler) recoverPage(w http.ResponseWriter, r *http.Request) {
	h.cfg.Web.Render(w, r, http.StatusOK, web.PageRecover, "Break-glass sign-in", web.RecoverPage{Code: r.URL.Query().Get("code")})
}

// recoverPost spends a recovery code and signs its owner in without SSO.
func (h *Handler) recoverPost(w http.ResponseWriter, r *http.Request) {
	if !session.SameOrigin(r) || !h.limiter.allow(r) {
		h.fail(w, r, http.StatusForbidden, "Request refused", "Open the recovery link again.")
		return
	}
	code := r.PostFormValue("code")
	var userID string
	err := db.Tx(r.Context(), h.cfg.Pool, func(tx pgx.Tx) error {
		id, err := consumeRecovery(r.Context(), tx, code)
		if err != nil {
			return err
		}
		userID = id
		e := audit.FromRequest(r, h.cfg.Proxies, nil)
		e.ActorType, e.ActorID = "user", id
		e.Action, e.TargetType, e.TargetID = "auth.recovery_login", "user", id
		e.Details = map[string]any{"amr": "recovery"}
		return audit.Append(r.Context(), tx, e)
	})
	if err != nil {
		h.record(r, "auth.recovery_login", "failure", nil, nil, map[string]any{"reason": reason(err)})
		h.cfg.Web.Render(w, r, http.StatusForbidden, web.PageRecover, "Break-glass sign-in",
			web.RecoverPage{Error: "This recovery link is invalid, expired or already used. Ask for a new one."})
		return
	}
	if _, err := h.cfg.Sessions.Create(r.Context(), w, r, session.Start{UserID: userID, AMR: []string{"recovery"}, IP: h.ip(r),
		UserAgent: r.UserAgent()}); err != nil {
		h.fail(w, r, http.StatusInternalServerError, "Sign-in failed", "Starlight could not start your session.")
		return
	}
	http.Redirect(w, r, h.cfg.DefaultReturn, http.StatusSeeOther)
}
