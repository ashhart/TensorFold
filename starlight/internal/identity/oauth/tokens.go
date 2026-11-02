package oauth

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Lifetimes bounds every credential the server issues.
type Lifetimes struct {
	Access          time.Duration
	RefreshIdle     time.Duration
	RefreshAbsolute time.Duration
}

// Issuer mints and rotates Starlight access and refresh tokens.
type Issuer struct {
	Pool *pgxpool.Pool
	Life Lifetimes
}

// Grant is what one successful token request returns.
type Grant struct {
	Access  string
	Refresh string
	Scope   string
	Expires time.Duration
}

// ErrInvalidGrant covers every refresh or code failure, deliberately without detail.
var ErrInvalidGrant = errors.New("invalid_grant")

// Issue creates an access token and, when offline_access was granted, a new refresh family.
func (i *Issuer) Issue(ctx context.Context, tx pgx.Tx, userID, clientID, scope, device string) (Grant, error) {
	g := Grant{Scope: scope, Expires: i.Life.Access}
	var family *string
	if hasScope(scope, discovery.ScopeOffline) {
		fid := ids.New()
		family = &fid
		tok, hash := secret.Generate(secret.Refresh)
		now := time.Now()
		if _, err := tx.Exec(ctx, `INSERT INTO refresh_tokens (id, family_id, token_hash, user_id, client_id, scope, device_name,
			family_expires_at, idle_expires_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
			ids.New(), fid, hash, userID, clientID, scope, device, now.Add(i.Life.RefreshAbsolute), now.Add(i.Life.RefreshIdle)); err != nil {
			return g, err
		}
		g.Refresh = tok
	}
	access, err := i.access(ctx, tx, userID, clientID, scope, family)
	g.Access = access
	return g, err
}

func (i *Issuer) access(ctx context.Context, tx pgx.Tx, userID, clientID, scope string, family *string) (string, error) {
	tok, hash := secret.Generate(secret.Access)
	_, err := tx.Exec(ctx, `INSERT INTO access_tokens (id, token_hash, user_id, family_id, client_id, scope, expires_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7)`, ids.New(), hash, userID, family, clientID, scope, time.Now().Add(i.Life.Access))
	return tok, err
}

// Rotate exchanges a refresh token for a new pair; presenting a spent token revokes the whole family.
func (i *Issuer) Rotate(ctx context.Context, refresh, clientID string) (Grant, string, error) {
	if k, ok := secret.Parse(refresh); !ok || k != secret.Refresh {
		return Grant{}, "", ErrInvalidGrant
	}
	var g Grant
	var userID string
	reused := false
	err := db.Tx(ctx, i.Pool, func(tx pgx.Tx) error {
		var id, family, client, scope, device, status string
		var used, revoked *time.Time
		var familyExp time.Time
		err := tx.QueryRow(ctx, `SELECT rt.id, rt.family_id, rt.user_id, rt.client_id, rt.scope, rt.device_name, rt.used_at, rt.revoked_at,
			rt.family_expires_at, u.status FROM refresh_tokens rt JOIN users u ON u.id = rt.user_id
			WHERE rt.token_hash = $1 AND rt.idle_expires_at > now() FOR UPDATE OF rt`, secret.Hash(refresh)).
			Scan(&id, &family, &userID, &client, &scope, &device, &used, &revoked, &familyExp, &status)
		if db.NotFound(err) {
			return ErrInvalidGrant
		}
		if err != nil {
			return err
		}
		if used != nil && revoked == nil {
			reused = true
			return revokeFamily(ctx, tx, family, "refresh_reuse")
		}
		if revoked != nil || client != clientID || status != "active" || time.Now().After(familyExp) {
			return ErrInvalidGrant
		}
		if _, err := tx.Exec(ctx, "UPDATE refresh_tokens SET used_at = now() WHERE id = $1", id); err != nil {
			return err
		}
		tok, hash := secret.Generate(secret.Refresh)
		idle := time.Now().Add(i.Life.RefreshIdle)
		if idle.After(familyExp) {
			idle = familyExp
		}
		if _, err := tx.Exec(ctx, `INSERT INTO refresh_tokens (id, family_id, token_hash, user_id, client_id, scope, device_name,
			family_expires_at, idle_expires_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
			ids.New(), family, hash, userID, client, scope, device, familyExp, idle); err != nil {
			return err
		}
		access, err := i.access(ctx, tx, userID, client, scope, &family)
		g = Grant{Access: access, Refresh: tok, Scope: scope, Expires: i.Life.Access}
		return err
	})
	if reused {
		return Grant{}, userID, ErrInvalidGrant
	}
	return g, userID, err
}

func revokeFamily(ctx context.Context, q db.Querier, family, reason string) error {
	if _, err := q.Exec(ctx, "UPDATE refresh_tokens SET revoked_at = now(), revoked_reason = $2 WHERE family_id = $1 AND revoked_at IS NULL",
		family, reason); err != nil {
		return err
	}
	_, err := q.Exec(ctx, "UPDATE access_tokens SET revoked_at = now() WHERE family_id = $1 AND revoked_at IS NULL", family)
	return err
}

// Revoke implements RFC 7009: a refresh token takes its family with it; unknown tokens are not an error.
func (i *Issuer) Revoke(ctx context.Context, token, clientID string) error {
	k, ok := secret.Parse(token)
	if !ok {
		return nil
	}
	switch k {
	case secret.Refresh:
		var family string
		err := i.Pool.QueryRow(ctx, "SELECT family_id FROM refresh_tokens WHERE token_hash = $1 AND client_id = $2", secret.Hash(token), clientID).Scan(&family)
		if db.NotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}
		return revokeFamily(ctx, i.Pool, family, "client_revoked")
	case secret.Access:
		_, err := i.Pool.Exec(ctx, "UPDATE access_tokens SET revoked_at = now() WHERE token_hash = $1 AND client_id = $2 AND revoked_at IS NULL",
			secret.Hash(token), clientID)
		return err
	}
	return nil
}
