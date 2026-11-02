// Package session issues browser sessions after SSO and guards them with idle, absolute and CSRF checks.
package session

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Config bounds session lifetime; Secure selects the __Host- cookie.
type Config struct {
	Idle     time.Duration
	Absolute time.Duration
	Secure   bool
}

// Manager creates and validates console sessions.
type Manager struct {
	pool   *pgxpool.Pool
	cfg    Config
	cookie string
}

// New returns a manager with defaults of 30 minutes idle and 12 hours absolute.
func New(pool *pgxpool.Pool, cfg Config) *Manager {
	if cfg.Idle <= 0 {
		cfg.Idle = 30 * time.Minute
	}
	if cfg.Absolute <= 0 {
		cfg.Absolute = 12 * time.Hour
	}
	name := "starlight_session"
	if cfg.Secure {
		name = "__Host-starlight_session"
	}
	return &Manager{pool: pool, cfg: cfg, cookie: name}
}

// Start describes a successful sign-in.
type Start struct {
	UserID       string
	ConnectionID string
	IDPSubject   string
	IDPSessionID string
	NameID       NameID
	AuthTime     time.Time
	AMR          []string
	IP           string
	UserAgent    string
}

// NameID holds the parts of a SAML NameID besides its value, which a LogoutRequest must repeat exactly.
type NameID struct {
	Format          string
	NameQualifier   string
	SPNameQualifier string
}

// Session is a validated console session; for SAML, IDPSubject is the NameID and IDPSessionID the SessionIndex.
type Session struct {
	ID           string
	UserID       string
	ConnectionID *string
	IDPSubject   string
	IDPSessionID string
	NameID       NameID
	AuthTime     time.Time
	AMR          []string
	CreatedAt    time.Time
	ExpiresAt    time.Time
	csrfKey      []byte
}

// ErrNoSession means the request carries no usable session.
var ErrNoSession = errors.New("session: none")

// ErrConnectionOff means the sign-in's connection was disabled or deleted before its session could start.
var ErrConnectionOff = errors.New("session: the sign-in's connection is disabled or gone")

// Create stores a session if its connection is still enabled, replaces any session cookie the browser had, and sets it.
func (m *Manager) Create(ctx context.Context, w http.ResponseWriter, r *http.Request, s Start) (Session, error) {
	if old, err := r.Cookie(m.cookie); err == nil {
		_, _ = m.pool.Exec(ctx, "UPDATE sessions SET revoked_at = now(), revoked_reason = 'replaced' WHERE token_hash = $1 AND revoked_at IS NULL",
			secret.Hash(old.Value))
	}
	token, hash := secret.Generate(secret.Session)
	now := time.Now()
	authTime := s.AuthTime
	if authTime.IsZero() || authTime.After(now) {
		authTime = now
	}
	var conn *string
	if s.ConnectionID != "" {
		conn = &s.ConnectionID
	}
	amr := s.AMR
	if amr == nil {
		amr = []string{}
	}
	sess := Session{ID: ids.New(), UserID: s.UserID, ConnectionID: conn, IDPSubject: s.IDPSubject, IDPSessionID: s.IDPSessionID, NameID: s.NameID,
		AuthTime: authTime, AMR: amr, CreatedAt: now, ExpiresAt: now.Add(m.cfg.Absolute), csrfKey: secret.Random(32)}
	err := db.Tx(ctx, m.pool, func(tx pgx.Tx) error {
		// FOR SHARE waits out a disable or delete in flight: one committed first refuses this session, a later one ends it.
		if conn != nil {
			var enabled bool
			err := tx.QueryRow(ctx, "SELECT enabled FROM idp_connections WHERE id = $1 FOR SHARE", *conn).Scan(&enabled)
			if db.NotFound(err) || (err == nil && !enabled) {
				return ErrConnectionOff
			}
			if err != nil {
				return err
			}
		}
		_, err := tx.Exec(ctx, `INSERT INTO sessions (id, token_hash, user_id, connection_id, idp_subject, idp_session_id, idp_name_format,
			idp_name_qualifier, idp_sp_name_qualifier, auth_time, amr, csrf_key, ip, user_agent, idle_expires_at, expires_at)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)`,
			sess.ID, hash, s.UserID, conn, s.IDPSubject, s.IDPSessionID, s.NameID.Format, s.NameID.NameQualifier, s.NameID.SPNameQualifier, authTime,
			amr, sess.csrfKey, s.IP, truncate(s.UserAgent, 256), now.Add(m.cfg.Idle), sess.ExpiresAt)
		return err
	})
	if err != nil {
		return Session{}, err
	}
	http.SetCookie(w, &http.Cookie{Name: m.cookie, Value: token, Path: "/", HttpOnly: true, Secure: m.cfg.Secure,
		SameSite: http.SameSiteLaxMode, Expires: sess.ExpiresAt})
	return sess, nil
}

// Load validates the request's session and its user, sliding the idle deadline at most once a minute.
func (m *Manager) Load(ctx context.Context, r *http.Request) (Session, directory.User, error) {
	c, err := r.Cookie(m.cookie)
	if err != nil {
		return Session{}, directory.User{}, ErrNoSession
	}
	if k, ok := secret.Parse(c.Value); !ok || k != secret.Session {
		return Session{}, directory.User{}, ErrNoSession
	}
	var s Session
	var lastSeen time.Time
	err = m.pool.QueryRow(ctx, `SELECT s.id, s.user_id, s.connection_id, s.idp_subject, s.idp_session_id, s.idp_name_format, s.idp_name_qualifier,
		s.idp_sp_name_qualifier, s.auth_time, s.amr, s.created_at, s.expires_at, s.csrf_key, s.last_seen_at
		FROM sessions s JOIN users u ON u.id = s.user_id
		WHERE s.token_hash = $1 AND s.revoked_at IS NULL AND s.expires_at > now() AND s.idle_expires_at > now() AND u.status = 'active'`,
		secret.Hash(c.Value)).Scan(&s.ID, &s.UserID, &s.ConnectionID, &s.IDPSubject, &s.IDPSessionID, &s.NameID.Format, &s.NameID.NameQualifier,
		&s.NameID.SPNameQualifier, &s.AuthTime, &s.AMR, &s.CreatedAt, &s.ExpiresAt, &s.csrfKey, &lastSeen)
	if db.NotFound(err) {
		return Session{}, directory.User{}, ErrNoSession
	}
	if err != nil {
		return Session{}, directory.User{}, err
	}
	if time.Since(lastSeen) > time.Minute {
		_, _ = m.pool.Exec(ctx, "UPDATE sessions SET last_seen_at = now(), idle_expires_at = LEAST(expires_at, now() + make_interval(secs => $2)) WHERE id = $1",
			s.ID, m.cfg.Idle.Seconds())
	}
	u, err := directory.GetUser(ctx, m.pool, s.UserID)
	return s, u, err
}

// Destroy revokes the request's session and clears the cookie.
func (m *Manager) Destroy(ctx context.Context, w http.ResponseWriter, r *http.Request) error {
	if c, err := r.Cookie(m.cookie); err == nil {
		if _, err := m.pool.Exec(ctx, "UPDATE sessions SET revoked_at = now(), revoked_reason = 'logout' WHERE token_hash = $1 AND revoked_at IS NULL",
			secret.Hash(c.Value)); err != nil {
			return err
		}
	}
	http.SetCookie(w, &http.Cookie{Name: m.cookie, Value: "", Path: "/", HttpOnly: true, Secure: m.cfg.Secure, SameSite: http.SameSiteLaxMode, MaxAge: -1})
	return nil
}

// CSRF returns the form token bound to this session.
func (s Session) CSRF() string {
	mac := hmac.New(sha256.New, s.csrfKey)
	mac.Write([]byte("csrf:" + s.ID))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

// ValidCSRF checks a submitted token in constant time.
func (s Session) ValidCSRF(token string) bool {
	return len(s.csrfKey) > 0 && hmac.Equal([]byte(token), []byte(s.CSRF()))
}

// Fresh reports whether the user authenticated at the IdP within maxAge, for sensitive actions.
func (s Session) Fresh(maxAge time.Duration) bool {
	return time.Since(s.AuthTime) <= maxAge
}

// RevokeIDP ends sessions named by an IdP logout: by IdP session id when given, else by subject.
func RevokeIDP(ctx context.Context, q db.Querier, connectionID, sid, subject string) (int64, error) {
	if sid == "" && subject == "" {
		return 0, errors.New("session: logout names neither session nor subject")
	}
	tag, err := q.Exec(ctx, `UPDATE sessions SET revoked_at = now(), revoked_reason = 'idp_logout'
		WHERE connection_id = $1 AND revoked_at IS NULL AND (($2 <> '' AND idp_session_id = $2) OR ($2 = '' AND idp_subject = $3))`,
		connectionID, sid, subject)
	return tag.RowsAffected(), err
}

// RevokeSAML ends a connection's sessions for a NameID, narrowed to the given SessionIndex values when there are any.
func RevokeSAML(ctx context.Context, q db.Querier, connectionID, nameID string, sessionIndexes []string) (int64, error) {
	if nameID == "" {
		return 0, errors.New("session: SAML logout names no NameID")
	}
	if sessionIndexes == nil {
		sessionIndexes = []string{}
	}
	tag, err := q.Exec(ctx, `UPDATE sessions SET revoked_at = now(), revoked_reason = 'idp_logout'
		WHERE connection_id = $1 AND revoked_at IS NULL AND idp_subject = $2 AND (cardinality($3::text[]) = 0 OR idp_session_id = ANY($3))`,
		connectionID, nameID, sessionIndexes)
	return tag.RowsAffected(), err
}

// RevokeConnection ends a connection's sessions and spends the OAuth codes they approved, in the caller's transaction.
func RevokeConnection(ctx context.Context, q db.Querier, connectionID, reason string) error {
	if _, err := q.Exec(ctx, `UPDATE oauth_codes SET used_at = now() WHERE used_at IS NULL
		AND session_id IN (SELECT id FROM sessions WHERE connection_id = $1 AND revoked_at IS NULL)`, connectionID); err != nil {
		return err
	}
	_, err := q.Exec(ctx, "UPDATE sessions SET revoked_at = now(), revoked_reason = $2 WHERE connection_id = $1 AND revoked_at IS NULL",
		connectionID, reason)
	return err
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
