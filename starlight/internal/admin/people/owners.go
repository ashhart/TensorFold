// Package people administers users, groups, role bindings, sessions and devices, auditing every change it makes.
package people

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// ErrLastOwner refuses a change that would leave the organisation without an active owner.
var ErrLastOwner = errors.New("this change would leave the organisation without an active owner")

// ErrNotFound means the record does not exist or does not belong to the caller.
var ErrNotFound = errors.New("people: not found")

// ownersLock serialises every change that can remove an owner, so two concurrent removals cannot both pass the check.
const ownersLock = 7314_0301

// activeOwnersSQL starts from the owner bindings, so its cost follows the number of owners rather than of users.
const activeOwnersSQL = `SELECT count(DISTINCT u.id) FROM users u JOIN (
	SELECT rb.subject_id AS user_id FROM role_bindings rb WHERE rb.role = 'owner' AND rb.project_id IS NULL AND rb.subject_type = 'user'
	UNION ALL SELECT gm.user_id FROM role_bindings rb JOIN group_members gm ON gm.group_id = rb.subject_id
		WHERE rb.role = 'owner' AND rb.project_id IS NULL AND rb.subject_type = 'group') o ON o.user_id = u.id
	WHERE u.status = 'active'`

// ActiveOwners counts active users who hold the organisation owner role directly or through a group.
func ActiveOwners(ctx context.Context, q db.Querier) (int, error) {
	var n int
	err := q.QueryRow(ctx, activeOwnersSQL).Scan(&n)
	return n, err
}

// IsOwner reports whether the user holds the organisation owner role directly or through a group.
func IsOwner(ctx context.Context, q db.Querier, userID string) (bool, error) {
	var ok bool
	err := q.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM role_bindings rb WHERE rb.role = 'owner' AND rb.project_id IS NULL
		AND rb.subject_type = 'user' AND rb.subject_id = $1) OR EXISTS (SELECT 1 FROM role_bindings rb JOIN group_members gm ON gm.group_id = rb.subject_id
		WHERE rb.role = 'owner' AND rb.project_id IS NULL AND rb.subject_type = 'group' AND gm.user_id = $1)`, userID).Scan(&ok)
	return ok, err
}

// GuardOwners locks owner changes and returns a check to run after the change, inside the same transaction.
func GuardOwners(ctx context.Context, tx pgx.Tx) (func() error, error) {
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", ownersLock); err != nil {
		return nil, err
	}
	before, err := ActiveOwners(ctx, tx)
	if err != nil {
		return nil, err
	}
	return func() error {
		after, err := ActiveOwners(ctx, tx)
		if err != nil {
			return err
		}
		if before > 0 && after == 0 {
			return ErrLastOwner
		}
		return nil
	}, nil
}
