package scim

import (
	"context"
	"errors"
	"net/http"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
)

// errLastOwner refuses a suspension or deletion that would leave the organisation without an active owner.
var errLastOwner = errf(http.StatusConflict, "",
	"this user is Starlight's last active owner; make someone else an owner in the Starlight console first")

// keepOwner makes a status change under the console's owner lock and refuses it when no active owner would remain.
func keepOwner(ctx context.Context, tx pgx.Tx, change func() error) error {
	check, err := people.GuardOwners(ctx, tx)
	if err != nil {
		return err
	}
	if err := change(); err != nil {
		return err
	}
	if err := check(); !errors.Is(err, people.ErrLastOwner) {
		return err
	}
	return errLastOwner
}

// refused audits an attempt keepOwner turned away, in its own transaction because the attempt's was rolled back.
func (h *Handler) refused(r *http.Request, action, userID, userName string) {
	e := h.event(r, action, "user", userID, map[string]any{"user_name": userName, "reason": "last_owner"})
	e.Outcome = "denied"
	_ = audit.Log(context.WithoutCancel(r.Context()), h.pool, e)
}
