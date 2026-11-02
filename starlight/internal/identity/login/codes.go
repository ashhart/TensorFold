package login

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// ownerLock serialises owner claims so two setup sign-ins cannot both become the first owner.
const ownerLock = 7314_0202

// MaxRecoveryTTL caps how long a break-glass link stays usable.
const MaxRecoveryTTL = 15 * time.Minute

// ErrOwnerExists means setup already finished; use a recovery code instead.
var ErrOwnerExists = errors.New("login: an owner already exists")

func ownerExists(ctx context.Context, q db.Querier) (bool, error) {
	n, err := people.ActiveOwners(ctx, q)
	return n > 0, err
}

// IssueSetupCode creates the one-time code that lets the first person to finish SSO become owner.
func IssueSetupCode(ctx context.Context, q db.Querier, ttl time.Duration) (string, error) {
	if exists, err := ownerExists(ctx, q); err != nil || exists {
		if err == nil {
			err = ErrOwnerExists
		}
		return "", err
	}
	if ttl <= 0 || ttl > 7*24*time.Hour {
		ttl = 24 * time.Hour
	}
	code, hash := secret.Generate(secret.Recovery)
	_, err := q.Exec(ctx, "INSERT INTO recovery_codes (id, code_hash, purpose, expires_at) VALUES ($1, $2, 'setup', $3)",
		ids.New(), hash, time.Now().Add(ttl))
	return code, err
}

// IssueRecoveryCode creates a break-glass code for an active owner, valid for at most fifteen minutes.
func IssueRecoveryCode(ctx context.Context, q db.Querier, ownerID string, ttl time.Duration) (string, error) {
	if ttl <= 0 || ttl > MaxRecoveryTTL {
		ttl = MaxRecoveryTTL
	}
	if err := activeOwner(ctx, q, ownerID); err != nil {
		return "", err
	}
	code, hash := secret.Generate(secret.Recovery)
	_, err := q.Exec(ctx, "INSERT INTO recovery_codes (id, code_hash, purpose, expires_at, user_id) VALUES ($1, $2, 'recovery', $3, $4)",
		ids.New(), hash, time.Now().Add(ttl), ownerID)
	return code, err
}

func activeOwner(ctx context.Context, q db.Querier, userID string) error {
	u, err := directory.GetUser(ctx, q, userID)
	if err != nil {
		return err
	}
	var owner bool
	if err := q.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM role_bindings WHERE role = 'owner' AND project_id IS NULL
		AND subject_type = 'user' AND subject_id = $1)`, userID).Scan(&owner); err != nil {
		return err
	}
	if u.Status != directory.Active || !owner {
		return errors.New("login: recovery codes are only issued to active owners")
	}
	return nil
}

// setupCode returns the id of a live setup code while no owner exists.
func setupCode(ctx context.Context, q db.Querier, code string) (string, error) {
	if k, ok := secret.Parse(code); !ok || k != secret.Recovery {
		return "", errSetupDone
	}
	if exists, err := ownerExists(ctx, q); err != nil || exists {
		if err == nil {
			err = errSetupDone
		}
		return "", err
	}
	var id string
	err := q.QueryRow(ctx, "SELECT id FROM recovery_codes WHERE code_hash = $1 AND purpose = 'setup' AND used_at IS NULL AND expires_at > now()",
		secret.Hash(code)).Scan(&id)
	if db.NotFound(err) {
		return "", errSetupDone
	}
	return id, err
}

// claimOwner spends this setup code and every other, and makes userID the first owner, inside the sign-in transaction.
func claimOwner(ctx context.Context, tx pgx.Tx, codeID, userID string) error {
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", ownerLock); err != nil {
		return err
	}
	if exists, err := ownerExists(ctx, tx); err != nil || exists {
		if err == nil {
			err = errSetupDone
		}
		return err
	}
	tag, err := tx.Exec(ctx, "UPDATE recovery_codes SET used_at = now(), used_by = $2 WHERE id = $1 AND purpose = 'setup' AND used_at IS NULL AND expires_at > now()",
		codeID, userID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return errSetupDone
	}
	// A spare link would otherwise reopen setup to whoever holds it, whenever no owner is active.
	if _, err := tx.Exec(ctx, "UPDATE recovery_codes SET used_at = now() WHERE purpose = 'setup' AND used_at IS NULL"); err != nil {
		return err
	}
	_, err = tx.Exec(ctx, "INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, 'owner', 'user', $2)", ids.New(), userID)
	return err
}

// consumeRecovery spends a recovery code and returns the owner it signs in.
func consumeRecovery(ctx context.Context, tx pgx.Tx, code string) (string, error) {
	if k, ok := secret.Parse(code); !ok || k != secret.Recovery {
		return "", errStale
	}
	var userID *string
	err := tx.QueryRow(ctx, `UPDATE recovery_codes SET used_at = now(), used_by = user_id
		WHERE code_hash = $1 AND purpose = 'recovery' AND used_at IS NULL AND expires_at > now() RETURNING user_id`, secret.Hash(code)).Scan(&userID)
	if db.NotFound(err) || (err == nil && userID == nil) {
		return "", errStale
	}
	if err != nil {
		return "", err
	}
	if err := activeOwner(ctx, tx, *userID); err != nil {
		return "", errStale
	}
	return *userID, nil
}

// Cleanup deletes expired sign-in and logout state; the server's janitor calls it periodically.
func Cleanup(ctx context.Context, q db.Querier) error {
	for _, stmt := range []string{
		"DELETE FROM auth_requests WHERE expires_at < now() - interval '1 day'",
		"DELETE FROM sso_replay WHERE expires_at < now()",
		"DELETE FROM saml_logout_requests WHERE expires_at < now()",
	} {
		if _, err := q.Exec(ctx, stmt); err != nil {
			return err
		}
	}
	return nil
}
