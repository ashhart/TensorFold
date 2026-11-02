// Package fleet is the service layer for projects, models, deployments, nodes, providers, limits and subscriptions.
package fleet

import (
	"context"
	"errors"
	"regexp"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

// Invalid is a correctable input error; its text is safe to show to the person who made the request.
type Invalid string

func (e Invalid) Error() string { return string(e) }

// ErrNotFound means the named record does not exist.
var ErrNotFound = errors.New("fleet: not found")

// Service applies fleet changes; Flush runs after every committed change so caches pick it up at once.
type Service struct {
	Pool  *pgxpool.Pool
	Keys  *keyring.Keyring
	Flush func()
}

// Tx is one unit of change; every write appends its audit event to the same transaction.
type Tx struct {
	tx    pgx.Tx
	actor audit.Event
	keys  *keyring.Keyring
	wrote bool
}

// Do runs f in one transaction attributed to actor, an audit event whose actor and request fields are filled.
func (s *Service) Do(ctx context.Context, actor audit.Event, f func(*Tx) error) error {
	var t *Tx
	err := db.Tx(ctx, s.Pool, func(tx pgx.Tx) error {
		t = &Tx{tx: tx, actor: actor, keys: s.Keys}
		return f(t)
	})
	if err == nil && t != nil && t.wrote && s.Flush != nil {
		s.Flush()
	}
	return err
}

// Wrote reports whether the transaction changed anything.
func (t *Tx) Wrote() bool { return t.wrote }

// Q exposes the transaction for read queries.
func (t *Tx) Q() pgx.Tx { return t.tx }

func (t *Tx) record(ctx context.Context, action, targetType, targetID string, details map[string]any) error {
	e := t.actor
	e.Action, e.TargetType, e.TargetID, e.Details, e.Outcome = action, targetType, targetID, details, "success"
	t.wrote = true
	return audit.Append(ctx, t.tx, e)
}

var (
	slugPattern  = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{1,62}$`)
	modelPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`)
	aliasPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$`)
	nodePattern  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$`)
	labelPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]{0,62}$`)
)

func slug(field, s string) error {
	if !slugPattern.MatchString(s) {
		return Invalid(field + " must be 2 to 63 lowercase letters, digits or hyphens, starting with a letter or digit")
	}
	return nil
}

func text(field, s string, max int, required bool) error {
	if required && strings.TrimSpace(s) == "" {
		return Invalid(field + " is required")
	}
	if len(s) > max {
		return Invalid(field + " is too long")
	}
	return nil
}

func class(s string) error {
	if _, err := levels.ParseClass(s); err != nil {
		return Invalid("classification must be public, internal, confidential or restricted")
	}
	return nil
}

func priority(s string) error {
	if _, err := levels.ParsePriority(s); err != nil {
		return Invalid("priority must be batch, standard or interactive")
	}
	return nil
}

func oneOf(field, s string, allowed ...string) error {
	for _, a := range allowed {
		if s == a {
			return nil
		}
	}
	return Invalid(field + " must be one of " + strings.Join(allowed, ", "))
}

// labels validates a key=value label map.
func labels(m map[string]string) error {
	if len(m) > 32 {
		return Invalid("at most 32 labels")
	}
	for k, v := range m {
		if !labelPattern.MatchString(k) || v != "" && !labelPattern.MatchString(v) {
			return Invalid("label " + k + " must be letters, digits, dots, dashes, slashes or underscores")
		}
	}
	return nil
}

func uniqueMsg(err error, msg string) error {
	if db.IsUnique(err, "") {
		return Invalid(msg)
	}
	return err
}

func optID(s string) *string {
	if ids.Valid(s) {
		return &s
	}
	return nil
}

func notFound(err error) error {
	if db.NotFound(err) {
		return ErrNotFound
	}
	return err
}
