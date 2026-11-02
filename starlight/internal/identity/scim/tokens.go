package scim

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// TokenInfo describes a SCIM token without its secret.
type TokenInfo struct {
	ID           string
	ConnectionID string
	Label        string
	CreatedAt    time.Time
	ExpiresAt    *time.Time
	LastUsedAt   *time.Time
	RevokedAt    *time.Time
}

// ErrTokenNotFound means no token with that id belongs to the connection.
var ErrTokenNotFound = errors.New("scim: token not found")

// IssueToken mints a provisioning token bound to one connection; the plaintext is returned once and never stored.
func IssueToken(ctx context.Context, q db.Querier, connectionID, label string, expiresAt *time.Time) (string, TokenInfo, error) {
	label = strings.TrimSpace(label)
	if len(label) > 64 {
		return "", TokenInfo{}, errors.New("scim: token label is longer than 64 characters")
	}
	if !ids.Valid(connectionID) {
		return "", TokenInfo{}, errors.New("scim: invalid connection id")
	}
	if expiresAt != nil && !expiresAt.After(time.Now()) {
		return "", TokenInfo{}, errors.New("scim: token expiry must be in the future")
	}
	token, hash := secret.Generate(secret.SCIM)
	info := TokenInfo{ID: ids.New(), ConnectionID: connectionID, Label: label, ExpiresAt: expiresAt}
	err := q.QueryRow(ctx, `INSERT INTO scim_tokens (id, connection_id, token_hash, label, expires_at) VALUES ($1, $2, $3, $4, $5)
		RETURNING created_at`, info.ID, connectionID, hash, label, expiresAt).Scan(&info.CreatedAt)
	if db.IsForeignKey(err) {
		return "", TokenInfo{}, errors.New("scim: connection does not exist")
	}
	if err != nil {
		return "", TokenInfo{}, err
	}
	return token, info, nil
}

// RevokeToken stops a connection's token from authenticating from the next request on.
func RevokeToken(ctx context.Context, q db.Querier, connectionID, tokenID string) error {
	if !ids.Valid(tokenID) || !ids.Valid(connectionID) {
		return ErrTokenNotFound
	}
	tag, err := q.Exec(ctx, "UPDATE scim_tokens SET revoked_at = coalesce(revoked_at, now()) WHERE id = $1 AND connection_id = $2",
		tokenID, connectionID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrTokenNotFound
	}
	return nil
}

// ListTokens returns a connection's tokens, newest first.
func ListTokens(ctx context.Context, q db.Querier, connectionID string) ([]TokenInfo, error) {
	rows, err := q.Query(ctx, `SELECT id, connection_id, label, created_at, expires_at, last_used_at, revoked_at
		FROM scim_tokens WHERE connection_id = $1 ORDER BY created_at DESC, id DESC`, connectionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []TokenInfo
	for rows.Next() {
		var t TokenInfo
		if err := rows.Scan(&t.ID, &t.ConnectionID, &t.Label, &t.CreatedAt, &t.ExpiresAt, &t.LastUsedAt, &t.RevokedAt); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}
