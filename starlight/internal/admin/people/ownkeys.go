package people

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

func keyDetails(k keys.Key) map[string]any {
	return map[string]any{"key": k.PublicID, "name": k.Name, "project_id": k.ProjectID, "priority": k.Priority}
}

// IssueOwnKey issues a key owned by the acting user; the secret is returned once.
func IssueOwnKey(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, s keys.Spec) (string, keys.Key, error) {
	var tok string
	var k keys.Key
	s.OwnerType, s.CreatedBy = "user", s.OwnerID
	err := db.Tx(ctx, pool, func(tx pgx.Tx) error {
		var err error
		if tok, k, err = keys.Issue(ctx, tx, s); err != nil {
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID, ev.Details = "key.issued", "api_key", k.ID, keyDetails(k)
		return audit.Append(ctx, tx, ev)
	})
	return tok, k, err
}

// ownKey loads a key and checks the user owns it.
func ownKey(ctx context.Context, q db.Querier, userID, keyID string) (keys.Key, error) {
	k, err := keys.Get(ctx, q, keyID)
	if err != nil || k.OwnerType != "user" || k.OwnerID != userID {
		return keys.Key{}, ErrNotFound
	}
	return k, nil
}

// RotateOwnKey replaces one of the user's keys, keeping the old one working for grace.
func RotateOwnKey(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, userID, keyID string, grace time.Duration) (string, keys.Key, error) {
	var tok string
	var k keys.Key
	err := db.Tx(ctx, pool, func(tx pgx.Tx) error {
		old, err := ownKey(ctx, tx, userID, keyID)
		if err != nil {
			return err
		}
		if tok, k, err = keys.Rotate(ctx, tx, old.ID, grace, userID); err != nil {
			return err
		}
		d := keyDetails(k)
		d["replaces"], d["grace_seconds"] = old.PublicID, int(grace.Seconds())
		ev.Action, ev.TargetType, ev.TargetID, ev.Details = "key.rotated", "api_key", k.ID, d
		return audit.Append(ctx, tx, ev)
	})
	return tok, k, err
}

// RevokeOwnKey revokes one of the user's keys immediately.
func RevokeOwnKey(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, userID, keyID string) error {
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		k, err := ownKey(ctx, tx, userID, keyID)
		if err != nil {
			return err
		}
		if err := keys.Revoke(ctx, tx, k.ID); err != nil {
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID, ev.Details = "key.revoked", "api_key", k.ID, keyDetails(k)
		return audit.Append(ctx, tx, ev)
	})
}
