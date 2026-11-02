package login

import (
	"errors"
	"log/slog"
	"net/http"
	"regexp"

	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
)

// errStale covers expired, replayed and unbound sign-in attempts alike, so none is distinguishable to a prober.
var errStale = errors.New("login: sign-in attempt expired, reused or not bound to this browser")

var errSetupDone = errors.New("login: setup link is invalid, expired or already used")

var oauthErrorCode = regexp.MustCompile(`^[a-z_]{1,64}$`)

type refusal struct {
	status  int
	heading string
	body    string
}

// explain turns a sign-in failure into a message the user can act on without leaking why a check failed.
func explain(err error) refusal {
	switch {
	case errors.Is(err, errStale):
		return refusal{http.StatusBadRequest, "Sign-in expired", "This sign-in attempt expired or was already used. Start again."}
	case errors.Is(err, errSetupDone):
		return refusal{http.StatusForbidden, "Setup link not valid", "This setup link is invalid, has expired, or setup is already complete."}
	case errors.Is(err, mapping.ErrNotProvisioned):
		return refusal{http.StatusForbidden, "Account not set up", "Your account has not been set up in Starlight yet. Ask your administrator for access."}
	case errors.Is(err, mapping.ErrDisabled):
		return refusal{http.StatusForbidden, "Account disabled", "Your Starlight account is suspended or deactivated."}
	case errors.Is(err, mapping.ErrTenant):
		return refusal{http.StatusForbidden, "Wrong organisation", "That account belongs to a different Microsoft Entra organisation."}
	case errors.Is(err, mapping.ErrHostedDomain):
		return refusal{http.StatusForbidden, "Wrong organisation", "Sign in with your work account on one of your organisation's domains."}
	case errors.Is(err, mapping.ErrUnverified):
		return refusal{http.StatusForbidden, "Email not verified", "Your identity provider has not verified your email address."}
	case errors.Is(err, mapping.ErrEmailUnverified), errors.Is(err, mapping.ErrIdentityConflict):
		return refusal{http.StatusForbidden, "Account conflict",
			"An account with this email already exists and could not be linked safely. Ask your administrator."}
	case errors.Is(err, mapping.ErrDomainNotAllowed):
		return refusal{http.StatusForbidden, "Domain not allowed", "Accounts from your email domain cannot use this sign-in method."}
	case errors.Is(err, mapping.ErrGroupRequired):
		return refusal{http.StatusForbidden, "Access not granted", "You are not in a group that is allowed to use Starlight."}
	case errors.Is(err, mapping.ErrMissingEmail), errors.Is(err, mapping.ErrMissingSubject):
		return refusal{http.StatusBadGateway, "Incomplete sign-in",
			"Your identity provider did not send the details Starlight needs. Ask your administrator to check the connection's claims."}
	case errors.Is(err, errGroups):
		return refusal{http.StatusBadGateway, "Groups unavailable", "Starlight could not read your group membership. Try again in a moment."}
	case errors.Is(err, session.ErrConnectionOff):
		return refusal{http.StatusForbidden, "Sign-in method turned off", "That sign-in method was turned off while you were signing in."}
	}
	return refusal{http.StatusUnauthorized, "Sign-in failed",
		"Starlight could not verify the response from your identity provider. Start again; if it keeps happening, ask your administrator."}
}

var errGroups = errors.New("login: directory group lookup failed")

func httpxLog(r *http.Request, msg string, err error) {
	slog.Warn(msg, "request_id", httpx.RequestID(r.Context()), "err", err)
}

func reason(err error) string {
	s := err.Error()
	if len(s) > 300 {
		s = s[:300]
	}
	return s
}
