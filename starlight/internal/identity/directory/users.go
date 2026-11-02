// Package directory stores users, their identities and groups, and revokes everything a user holds.
package directory

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// User statuses.
const (
	Active      = "active"
	Suspended   = "suspended"
	Deactivated = "deactivated"
)

// User is one person known to Starlight.
type User struct {
	ID               string
	Email            string
	DisplayName      string
	Status           string
	Source           string
	SCIMConnectionID *string
	SCIMUserName     *string
	ExternalID       *string
	CreatedAt        time.Time
	UpdatedAt        time.Time
	LastLoginAt      *time.Time
}

// ErrNotFound means no matching user, identity or group.
var ErrNotFound = errors.New("directory: not found")

// ErrConflict means a live user already holds the email or SCIM user name.
var ErrConflict = errors.New("directory: conflict")

const userCols = `id, email, display_name, status, source, scim_connection_id, scim_user_name, external_id, created_at, updated_at, last_login_at`

func scanUser(row pgx.Row) (User, error) {
	var u User
	err := row.Scan(&u.ID, &u.Email, &u.DisplayName, &u.Status, &u.Source, &u.SCIMConnectionID, &u.SCIMUserName, &u.ExternalID,
		&u.CreatedAt, &u.UpdatedAt, &u.LastLoginAt)
	if db.NotFound(err) {
		return u, ErrNotFound
	}
	return u, err
}

// GetUser loads a user by id.
func GetUser(ctx context.Context, q db.Querier, id string) (User, error) {
	if !ids.Valid(id) {
		return User{}, ErrNotFound
	}
	return scanUser(q.QueryRow(ctx, "SELECT "+userCols+" FROM users WHERE id = $1", id))
}

// FindByEmail loads the live (not deactivated) user with this email, case-insensitively.
func FindByEmail(ctx context.Context, q db.Querier, email string) (User, error) {
	return scanUser(q.QueryRow(ctx, "SELECT "+userCols+" FROM users WHERE email_lower = lower($1) AND status <> 'deactivated'", email))
}

// NewUser is the input for CreateUser.
type NewUser struct {
	Email            string
	DisplayName      string
	Source           string
	SCIMConnectionID *string
	SCIMUserName     *string
	ExternalID       *string
	Status           string
}

// CreateUser inserts a user and reports ErrConflict when the email or SCIM name is taken.
func CreateUser(ctx context.Context, q db.Querier, n NewUser) (User, error) {
	email := strings.TrimSpace(n.Email)
	if email == "" || !strings.Contains(email, "@") || len(email) > 320 {
		return User{}, errors.New("directory: invalid email")
	}
	status := n.Status
	if status == "" {
		status = Active
	}
	u, err := scanUser(q.QueryRow(ctx, `INSERT INTO users (id, email, display_name, status, source, scim_connection_id, scim_user_name, external_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING `+userCols,
		ids.New(), email, strings.TrimSpace(n.DisplayName), status, n.Source, n.SCIMConnectionID, n.SCIMUserName, n.ExternalID))
	if db.IsUnique(err, "") {
		return User{}, ErrConflict
	}
	return u, err
}

// UpdateProfile changes the name and email; email changes keep the user id and every grant.
func UpdateProfile(ctx context.Context, q db.Querier, id, email, displayName string) error {
	tag, err := q.Exec(ctx, "UPDATE users SET email = $2, display_name = $3, updated_at = now() WHERE id = $1", id, email, displayName)
	if db.IsUnique(err, "") {
		return ErrConflict
	}
	if err == nil && tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return err
}

// SetStatus moves a user between statuses; leaving active revokes every credential at once.
func SetStatus(ctx context.Context, tx pgx.Tx, id, status, reason string) error {
	switch status {
	case Active, Suspended, Deactivated:
	default:
		return errors.New("directory: invalid status")
	}
	tag, err := tx.Exec(ctx, `UPDATE users SET status = $2, updated_at = now(),
		deactivated_at = CASE WHEN $2 = 'deactivated' THEN now() ELSE NULL END WHERE id = $1`, id, status)
	if db.IsUnique(err, "") {
		return ErrConflict
	}
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	if status != Active {
		return RevokeUser(ctx, tx, id, reason)
	}
	return nil
}

// RevokeUser ends every session and token the user holds and revokes keys and tasks they own.
func RevokeUser(ctx context.Context, q db.Querier, userID, reason string) error {
	if reason == "" {
		reason = "revoked"
	}
	if _, err := q.Exec(ctx, "UPDATE sessions SET revoked_at = now(), revoked_reason = $2 WHERE user_id = $1 AND revoked_at IS NULL",
		userID, reason); err != nil {
		return err
	}
	if err := RevokeOAuth(ctx, q, []string{userID}, reason); err != nil {
		return err
	}
	plain := []string{
		"UPDATE api_keys SET revoked_at = now() WHERE owner_type = 'user' AND owner_id = $1 AND revoked_at IS NULL",
		"UPDATE tasks SET revoked_at = now() WHERE owner_type = 'user' AND owner_id = $1 AND revoked_at IS NULL",
	}
	for _, s := range plain {
		if _, err := q.Exec(ctx, s, userID); err != nil {
			return err
		}
	}
	return nil
}

// RevokeOAuth ends what the users' sign-ins granted: refresh families, access tokens, device grants, codes and tasks.
func RevokeOAuth(ctx context.Context, q db.Querier, userIDs []string, reason string) error {
	if _, err := q.Exec(ctx, "UPDATE refresh_tokens SET revoked_at = now(), revoked_reason = $2 WHERE user_id = ANY($1::uuid[]) AND revoked_at IS NULL",
		userIDs, reason); err != nil {
		return err
	}
	for _, s := range []string{
		"UPDATE access_tokens SET revoked_at = now() WHERE user_id = ANY($1::uuid[]) AND revoked_at IS NULL",
		"UPDATE device_grants SET status = 'denied' WHERE user_id = ANY($1::uuid[]) AND status IN ('pending', 'approved')",
		"UPDATE oauth_codes SET used_at = now() WHERE user_id = ANY($1::uuid[]) AND used_at IS NULL",
		"UPDATE tasks SET revoked_at = now() WHERE owner_type = 'user' AND owner_id = ANY($1::uuid[]) AND api_key_id IS NULL AND revoked_at IS NULL",
	} {
		if _, err := q.Exec(ctx, s, userIDs); err != nil {
			return err
		}
	}
	return nil
}

// TouchLogin records a successful sign-in.
func TouchLogin(ctx context.Context, q db.Querier, userID string) error {
	_, err := q.Exec(ctx, "UPDATE users SET last_login_at = now() WHERE id = $1", userID)
	return err
}

// FindIdentity returns the user linked to a subject at a connection.
func FindIdentity(ctx context.Context, q db.Querier, connectionID, subject string) (string, error) {
	var userID string
	err := q.QueryRow(ctx, "SELECT user_id FROM identities WHERE connection_id = $1 AND subject = $2", connectionID, subject).Scan(&userID)
	if db.NotFound(err) {
		return "", ErrNotFound
	}
	return userID, err
}

// LinkIdentity binds a subject to a user, refreshing the email seen at login.
func LinkIdentity(ctx context.Context, q db.Querier, userID, connectionID, subject, email string) error {
	_, err := q.Exec(ctx, `INSERT INTO identities (id, user_id, connection_id, subject, email, last_login_at)
		VALUES ($1, $2, $3, $4, $5, now())
		ON CONFLICT (connection_id, subject) DO UPDATE SET email = EXCLUDED.email, last_login_at = now()
		WHERE identities.user_id = EXCLUDED.user_id`, ids.New(), userID, connectionID, subject, email)
	return err
}
