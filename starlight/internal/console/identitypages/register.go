// Package identitypages serves the console pages for accounts, sign-on, users, groups, access, audit and settings.
package identitypages

import (
	"context"
	"embed"
	"errors"
	"net/http"
	"strconv"

	"github.com/jackc/pgx/v5/pgconn"

	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/admin/settings"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/scim"
)

//go:embed templates
var templateFS embed.FS

// Templates holds this package's pages under templates/identity/.
var Templates = templateFS

// pages carries the console to every handler.
type pages struct{ c *console.Console }

// Register mounts every identity page on the console.
func Register(c *console.Console) {
	p := pages{c: c}
	p.account()
	p.sso()
	p.users()
	p.groups()
	p.access()
	p.auditLog()
	p.settings()
}

// friendly turns service errors into messages the console shows the administrator.
func friendly(err error) error {
	var inv settings.ErrInvalid
	var ve *connections.ValidationError
	switch {
	case err == nil:
		return nil
	case errors.Is(err, people.ErrLastOwner), errors.Is(err, people.ErrExists), errors.Is(err, people.ErrRole):
		return console.Invalid(capital(err.Error()) + ".")
	case errors.Is(err, people.ErrNotFound), errors.Is(err, keys.ErrNotFound), errors.Is(err, connections.ErrNotFound),
		errors.Is(err, scim.ErrTokenNotFound), errors.Is(err, directory.ErrNotFound):
		return httpx.Errorf(http.StatusNotFound, "not_found", "That record does not exist.")
	case errors.Is(err, directory.ErrConflict):
		return console.Invalid("Another active user already has this email address.")
	case errors.As(err, &inv):
		return console.Invalid(capital(string(inv)) + ".")
	case errors.As(err, &ve):
		return console.Invalid(ve.Field + ": " + ve.Message + ".")
	case errors.Is(err, connections.ErrSlugTaken):
		return console.Invalid("That slug is already used by another connection.")
	case errors.Is(err, connections.ErrDomainTaken):
		return console.Invalid("Another connection already routes one of those email domains.")
	}
	return err
}

// asInvalid shows a service's validation message to the administrator but keeps database failures opaque.
func asInvalid(err error) error {
	if fe := friendly(err); fe != err {
		return fe
	}
	var pg *pgconn.PgError
	if errors.As(err, &pg) || errors.Is(err, context.Canceled) {
		return err
	}
	return console.Invalid(capital(err.Error()) + ".")
}

func capital(s string) string {
	if s == "" {
		return s
	}
	b := []byte(s)
	if b[0] >= 'a' && b[0] <= 'z' {
		b[0] -= 'a' - 'A'
	}
	return string(b)
}

// requireOwnerRights refuses actions on owners for administrators who cannot manage owners.
func requireOwnerRights(q *console.Request, owner bool) error {
	if owner && !q.Can(authz.OwnersWrite, "") {
		return httpx.Errorf(http.StatusForbidden, "forbidden", "Only owners can change another owner.")
	}
	return nil
}

func atoi(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}
