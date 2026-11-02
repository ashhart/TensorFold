package connections

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// writeLock serialises connection writes so domain routing stays unambiguous.
const writeLock = 7314_0201

// ErrNotFound means no connection matched.
var ErrNotFound = errors.New("connections: not found")

// ErrSlugTaken means another connection already uses the slug.
var ErrSlugTaken = errors.New("connections: slug already in use")

// Store reads and writes connections, sealing secrets with the keyring.
type Store struct {
	pool *pgxpool.Pool
	kr   *keyring.Keyring
}

// NewStore returns a store over pool and kr.
func NewStore(pool *pgxpool.Pool, kr *keyring.Keyring) *Store {
	return &Store{pool: pool, kr: kr}
}

const cols = `id, slug, name, protocol, preset, config, secrets IS NOT NULL, domains, enabled, jit, created_at, updated_at`

func scan(row pgx.Row) (Connection, error) {
	var c Connection
	var raw []byte
	err := row.Scan(&c.ID, &c.Slug, &c.Name, &c.Protocol, &c.Preset, &raw, &c.HasSecret, &c.Domains, &c.Enabled, &c.JIT, &c.CreatedAt, &c.UpdatedAt)
	if db.NotFound(err) {
		return c, ErrNotFound
	}
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(raw, &c.Config); err != nil {
		return c, err
	}
	return c, nil
}

func aad(id string) string { return "idp_connections.secrets:" + id }

// Get loads a connection by id.
func (s *Store) Get(ctx context.Context, id string) (Connection, error) {
	if !ids.Valid(id) {
		return Connection{}, ErrNotFound
	}
	return scan(s.pool.QueryRow(ctx, "SELECT "+cols+" FROM idp_connections WHERE id = $1", id))
}

// GetBySlug loads a connection by its URL slug.
func (s *Store) GetBySlug(ctx context.Context, slug string) (Connection, error) {
	return scan(s.pool.QueryRow(ctx, "SELECT "+cols+" FROM idp_connections WHERE slug = $1", slug))
}

// List returns every connection ordered by name.
func (s *Store) List(ctx context.Context) ([]Connection, error) {
	rows, err := s.pool.Query(ctx, "SELECT "+cols+" FROM idp_connections ORDER BY name, slug")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Connection
	for rows.Next() {
		c, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// Options lists the enabled connections as login-page buttons.
func (s *Store) Options(ctx context.Context) ([]Option, error) {
	all, err := s.List(ctx)
	if err != nil {
		return nil, err
	}
	var out []Option
	for _, c := range all {
		if c.Enabled {
			out = append(out, Option{Slug: c.Slug, Name: c.Name, Preset: c.Preset, Protocol: c.Protocol})
		}
	}
	return out, nil
}

// ForEmail finds the enabled connection that routes the email's domain.
func (s *Store) ForEmail(ctx context.Context, email string) (Connection, error) {
	_, domain, ok := strings.Cut(strings.ToLower(strings.TrimSpace(email)), "@")
	if !ok || domain == "" {
		return Connection{}, ErrNotFound
	}
	return scan(s.pool.QueryRow(ctx, "SELECT "+cols+" FROM idp_connections WHERE enabled AND $1 = ANY(domains) ORDER BY created_at LIMIT 1", domain))
}

// Secrets opens a connection's sealed secrets.
func (s *Store) Secrets(ctx context.Context, c Connection) (Secrets, error) {
	var sealed []byte
	if err := s.pool.QueryRow(ctx, "SELECT secrets FROM idp_connections WHERE id = $1", c.ID).Scan(&sealed); err != nil {
		if db.NotFound(err) {
			return Secrets{}, ErrNotFound
		}
		return Secrets{}, err
	}
	return s.open(c.ID, sealed)
}

func (s *Store) open(id string, sealed []byte) (Secrets, error) {
	var out Secrets
	if len(sealed) == 0 {
		return out, nil
	}
	plain, err := s.kr.Open(sealed, aad(id))
	if err != nil {
		return out, err
	}
	err = json.Unmarshal(plain, &out)
	return out, err
}

func (s *Store) seal(id string, sec Secrets) ([]byte, error) {
	if sec == (Secrets{}) {
		return nil, nil
	}
	plain, err := json.Marshal(sec)
	if err != nil {
		return nil, err
	}
	return s.kr.Seal(plain, aad(id))
}

// Recorder audits a change inside the transaction that makes it, so no change commits unaudited; nil skips it.
type Recorder func(tx pgx.Tx, c Connection) error

func record(rec Recorder, tx pgx.Tx, c Connection, err error) error {
	if err != nil || rec == nil {
		return err
	}
	return rec(tx, c)
}

// Create validates and stores a new connection, generating a SAML signing pair when none is given.
func (s *Store) Create(ctx context.Context, in Input, rec Recorder) (Connection, error) {
	if err := in.Validate(); err != nil {
		return Connection{}, err
	}
	id := ids.New()
	if err := checkClientKey(in); err != nil {
		return Connection{}, err
	}
	if err := ensureSPKeys(&in, Secrets{}); err != nil {
		return Connection{}, err
	}
	var out Connection
	err := db.Tx(ctx, s.pool, func(tx pgx.Tx) error {
		if err := lockAndCheckDomains(ctx, tx, id, in.Domains); err != nil {
			return err
		}
		cfg, err := json.Marshal(in.Config)
		if err != nil {
			return err
		}
		sealed, err := s.seal(id, in.Secrets)
		if err != nil {
			return err
		}
		out, err = scan(tx.QueryRow(ctx, `INSERT INTO idp_connections (id, slug, name, protocol, preset, config, secrets, domains, enabled, jit)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING `+cols,
			id, in.Slug, in.Name, in.Protocol, in.Preset, cfg, sealed, in.Domains, in.Enabled, in.JIT))
		if db.IsUnique(err, "idp_connections_slug_key") {
			return ErrSlugTaken
		}
		return record(rec, tx, out, err)
	})
	return out, err
}

// Update replaces a connection's settings; empty secret fields keep stored values; disabling it signs its users out.
func (s *Store) Update(ctx context.Context, id string, in Input, rec Recorder) (Connection, error) {
	if err := in.Validate(); err != nil {
		return Connection{}, err
	}
	cur, err := s.Get(ctx, id)
	if err != nil {
		return Connection{}, err
	}
	old, err := s.Secrets(ctx, cur)
	if err != nil {
		return Connection{}, err
	}
	if o := in.Config.OIDC; o != nil && cur.Config.OIDC != nil && o.SigningCertificate == "" && in.Secrets.SigningKeyPEM == "" {
		o.SigningCertificate = cur.Config.OIDC.SigningCertificate
	}
	in.Secrets = old.Merge(in.Secrets)
	if err := checkClientKey(in); err != nil {
		return Connection{}, err
	}
	if in.Config.SAML != nil && cur.Config.SAML != nil && in.Config.SAML.SPCertificate == "" {
		in.Config.SAML.SPCertificate = cur.Config.SAML.SPCertificate
	}
	if err := ensureSPKeys(&in, old); err != nil {
		return Connection{}, err
	}
	var out Connection
	err = db.Tx(ctx, s.pool, func(tx pgx.Tx) error {
		if err := lockAndCheckDomains(ctx, tx, id, in.Domains); err != nil {
			return err
		}
		cfg, err := json.Marshal(in.Config)
		if err != nil {
			return err
		}
		sealed, err := s.seal(id, in.Secrets)
		if err != nil {
			return err
		}
		out, err = scan(tx.QueryRow(ctx, `UPDATE idp_connections SET slug = $2, name = $3, protocol = $4, preset = $5, config = $6, secrets = $7,
			domains = $8, enabled = $9, jit = $10, updated_at = now() WHERE id = $1 RETURNING `+cols,
			id, in.Slug, in.Name, in.Protocol, in.Preset, cfg, sealed, in.Domains, in.Enabled, in.JIT))
		if db.IsUnique(err, "idp_connections_slug_key") {
			return ErrSlugTaken
		}
		if err == nil && cur.Enabled && !out.Enabled {
			err = signOut(ctx, tx, id, "connection_disabled")
		}
		return record(rec, tx, out, err)
	})
	return out, err
}

// SetEnabled turns a connection on or off without touching its settings; turning it off signs its users out.
func (s *Store) SetEnabled(ctx context.Context, id string, enabled bool, rec Recorder) error {
	return db.Tx(ctx, s.pool, func(tx pgx.Tx) error {
		c, err := scan(tx.QueryRow(ctx, "UPDATE idp_connections SET enabled = $2, updated_at = now() WHERE id = $1 RETURNING "+cols, id, enabled))
		if err == nil && !enabled {
			err = signOut(ctx, tx, id, "connection_disabled")
		}
		return record(rec, tx, c, err)
	})
}

// Delete signs out a connection's users, then removes it with its identities and groups unless no active owner is left.
func (s *Store) Delete(ctx context.Context, id string, rec Recorder) error {
	return db.Tx(ctx, s.pool, func(tx pgx.Tx) error {
		keepOwner, err := people.GuardOwners(ctx, tx)
		if err != nil {
			return err
		}
		// Locking the row first makes a sign-in finishing now either land before the sign-out or find the connection gone.
		if _, err := tx.Exec(ctx, "SELECT 1 FROM idp_connections WHERE id = $1 FOR UPDATE", id); err != nil {
			return err
		}
		if err := signOut(ctx, tx, id, "connection_deleted"); err != nil {
			return err
		}
		c, err := scan(tx.QueryRow(ctx, "DELETE FROM idp_connections WHERE id = $1 RETURNING "+cols, id))
		if err == nil {
			err = keepOwner()
		}
		return record(rec, tx, c, err)
	})
}

// signOut ends a connection's sessions and the Starlight OAuth grants of every user with an identity through it.
func signOut(ctx context.Context, tx pgx.Tx, id, reason string) error {
	var users []string
	if err := tx.QueryRow(ctx, "SELECT array_agg(DISTINCT user_id) FROM identities WHERE connection_id = $1", id).Scan(&users); err != nil {
		return err
	}
	if err := session.RevokeConnection(ctx, tx, id, reason); err != nil {
		return err
	}
	return directory.RevokeOAuth(ctx, tx, users, reason)
}

func lockAndCheckDomains(ctx context.Context, tx pgx.Tx, id string, domains []string) error {
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", writeLock); err != nil {
		return err
	}
	if len(domains) == 0 {
		return nil
	}
	var other string
	err := tx.QueryRow(ctx, "SELECT slug FROM idp_connections WHERE id <> $1 AND domains && $2::text[] LIMIT 1", id, domains).Scan(&other)
	if db.NotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	return ErrDomainTaken
}
