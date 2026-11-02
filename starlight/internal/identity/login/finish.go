package login

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
)

// finish reads directory groups when needed, resolves the user, claims ownership for setup, then opens a session.
func (h *Handler) finish(w http.ResponseWriter, r *http.Request, conn connections.Connection, secrets connections.Secrets, ar authRequest,
	id mapping.Identity, nameID session.NameID) {
	ctx := r.Context()
	if err := h.enrich(ctx, conn, secrets, &id); err != nil {
		h.refuse(w, r, &conn, "groups", fmt.Errorf("%w: %v", errGroups, err))
		return
	}
	var user directory.User
	err := db.Tx(ctx, h.cfg.Pool, func(tx pgx.Tx) error {
		u, err := mapping.Resolve(ctx, tx, conn, id)
		if err != nil {
			return err
		}
		user = u
		details := map[string]any{"connection": conn.Slug, "groups": len(id.Groups)}
		if ar.SetupCodeID != nil {
			if err := claimOwner(ctx, tx, *ar.SetupCodeID, u.ID); err != nil {
				return err
			}
			e := audit.FromRequest(r, h.cfg.Proxies, nil)
			e.ActorType, e.ActorID, e.ActorLabel = "user", u.ID, u.Email
			e.Action, e.TargetType, e.TargetID = "setup.owner_claimed", "user", u.ID
			if err := audit.Append(ctx, tx, e); err != nil {
				return err
			}
		}
		e := audit.FromRequest(r, h.cfg.Proxies, nil)
		e.ActorType, e.ActorID, e.ActorLabel = "user", u.ID, u.Email
		e.Action, e.TargetType, e.TargetID, e.Details = "auth.login", "idp_connection", conn.ID, details
		return audit.Append(ctx, tx, e)
	})
	if err != nil {
		h.refuse(w, r, &conn, "resolve", err)
		return
	}
	amr := id.AMR
	if amr == nil {
		amr = []string{}
	}
	_, err = h.cfg.Sessions.Create(ctx, w, r, session.Start{UserID: user.ID, ConnectionID: conn.ID, IDPSubject: id.RawSubject,
		IDPSessionID: id.SessionID, NameID: nameID, AuthTime: id.AuthTime, AMR: amr, IP: h.ip(r), UserAgent: r.UserAgent()})
	if err != nil {
		h.refuse(w, r, &conn, "session", err)
		return
	}
	http.Redirect(w, r, session.SafeReturn(ar.ReturnTo, h.cfg.DefaultReturn), http.StatusSeeOther)
}

// enrich replaces claim groups with directory groups where the connection's strategy needs an API call.
func (h *Handler) enrich(ctx context.Context, conn connections.Connection, secrets connections.Secrets, id *mapping.Identity) error {
	switch conn.Config.Groups.Strategy {
	case connections.GroupsEntraGraph:
		if !id.Overage {
			return nil
		}
		tenant, oid, ok := strings.Cut(id.Subject, ":")
		if !ok {
			return fmt.Errorf("entra subject %q is not tenant:object", id.Subject)
		}
		cred, err := graphCredential(conn, secrets, h.cfg.Now)
		if err != nil {
			return err
		}
		groups, err := h.cfg.Entra.Groups(ctx, tenant, cred, oid)
		if err != nil {
			return err
		}
		id.Groups, id.GroupsKnown = groups, true
	case connections.GroupsGoogleDirectory:
		groups, err := h.cfg.Google.Groups(ctx, secrets.GoogleServiceAccount, conn.Config.Provider.GoogleAdminEmail, id.Email,
			conn.Config.Groups.NestedDepth)
		if err != nil {
			return err
		}
		id.Groups, id.GroupsKnown = groups, true
	default:
		if id.Overage {
			id.GroupsKnown = false
		}
	}
	return nil
}

// graphCredential is the connection's certificate when it signs in with one, else the Graph or sign-in client secret.
func graphCredential(conn connections.Connection, secrets connections.Secrets, now func() time.Time) (*oidc.Credential, error) {
	clientID, o := conn.Config.Provider.GraphClientID, conn.Config.OIDC
	if clientID == "" && o != nil {
		clientID = o.ClientID
	}
	if o != nil && o.AuthMethod == connections.AuthPrivateJWT {
		return oidc.KeyCredential(clientID, secrets.SigningKeyPEM, o.SigningCertificate, o.SigningKeyID, now)
	}
	secret := secrets.GraphClientSecret
	if secret == "" {
		secret = secrets.ClientSecret
	}
	return oidc.SecretCredential(clientID, secret), nil
}
