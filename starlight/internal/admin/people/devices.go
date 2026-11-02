package people

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Device is one Connect or CLI sign-in: a refresh-token family and its access tokens.
type Device struct {
	FamilyID   string
	ClientID   string
	DeviceName string
	Created    time.Time
	LastUsed   time.Time
	Live       bool
}

// Devices lists a user's refresh-token families, newest activity first.
func Devices(ctx context.Context, q db.Querier, userID string) ([]Device, error) {
	rows, err := q.Query(ctx, `SELECT family_id, min(client_id), max(device_name), min(created_at), max(coalesce(used_at, created_at)),
		bool_or(revoked_at IS NULL AND used_at IS NULL AND idle_expires_at > now() AND family_expires_at > now())
		FROM refresh_tokens WHERE user_id = $1 GROUP BY family_id ORDER BY 5 DESC LIMIT 200`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Device
	for rows.Next() {
		var d Device
		if err := rows.Scan(&d.FamilyID, &d.ClientID, &d.DeviceName, &d.Created, &d.LastUsed, &d.Live); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// RevokeDevice ends one of the user's refresh-token families and every access token issued from it.
func RevokeDevice(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, userID, familyID string) error {
	if !ids.Valid(familyID) {
		return ErrNotFound
	}
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		var device, client string
		err := tx.QueryRow(ctx, "SELECT max(device_name), min(client_id) FROM refresh_tokens WHERE family_id = $1 AND user_id = $2 HAVING count(*) > 0",
			familyID, userID).Scan(&device, &client)
		if db.NotFound(err) {
			return ErrNotFound
		}
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE refresh_tokens SET revoked_at = now(), revoked_reason = 'user_revoked'
			WHERE family_id = $1 AND user_id = $2 AND revoked_at IS NULL`, familyID, userID); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, "UPDATE access_tokens SET revoked_at = now() WHERE family_id = $1 AND user_id = $2 AND revoked_at IS NULL",
			familyID, userID); err != nil {
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID = "device.revoked", "user", userID
		ev.Details = map[string]any{"family_id": familyID, "client": client, "device": device}
		return audit.Append(ctx, tx, ev)
	})
}

// Session is one live console session.
type Session struct {
	ID         string
	Created    time.Time
	LastSeen   time.Time
	IP         string
	UserAgent  string
	Connection string
	AMR        []string
}

// Sessions lists the user's live console sessions.
func Sessions(ctx context.Context, q db.Querier, userID string) ([]Session, error) {
	rows, err := q.Query(ctx, `SELECT s.id, s.created_at, s.last_seen_at, s.ip, s.user_agent, coalesce(c.name, ''), s.amr FROM sessions s
		LEFT JOIN idp_connections c ON c.id = s.connection_id
		WHERE s.user_id = $1 AND s.revoked_at IS NULL AND s.expires_at > now() AND s.idle_expires_at > now() ORDER BY s.last_seen_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Session
	for rows.Next() {
		var s Session
		if err := rows.Scan(&s.ID, &s.Created, &s.LastSeen, &s.IP, &s.UserAgent, &s.Connection, &s.AMR); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// RevokeSession signs one of the user's console sessions out.
func RevokeSession(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, userID, sessionID string) error {
	if !ids.Valid(sessionID) {
		return ErrNotFound
	}
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		tag, err := tx.Exec(ctx, `UPDATE sessions SET revoked_at = now(), revoked_reason = 'user_revoked'
			WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL`, sessionID, userID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return ErrNotFound
		}
		ev.Action, ev.TargetType, ev.TargetID = "session.revoked", "user", userID
		ev.Details = map[string]any{"session_id": sessionID}
		return audit.Append(ctx, tx, ev)
	})
}

// Task is one root agent task with a count of the subtasks under it.
type Task struct {
	ID        string
	Name      string
	Project   string
	Models    []string
	LocalOnly bool
	Children  int
	ExpiresAt time.Time
	Revoked   bool
	CreatedAt time.Time
}

// Live reports whether the task can still authenticate.
func (t Task) Live() bool { return !t.Revoked && t.ExpiresAt.After(time.Now()) }

// Tasks lists root tasks the user owns, newest first.
func Tasks(ctx context.Context, q db.Querier, userID string) ([]Task, error) {
	rows, err := q.Query(ctx, `SELECT t.id, t.name, p.slug, t.models, t.local_only,
		(SELECT count(*) FROM tasks c WHERE c.root_id = t.id AND c.id <> t.id), t.expires_at, t.revoked_at IS NOT NULL, t.created_at
		FROM tasks t JOIN projects p ON p.id = t.project_id
		WHERE t.owner_type = 'user' AND t.owner_id = $1 AND t.parent_id IS NULL ORDER BY t.created_at DESC LIMIT 100`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Task
	for rows.Next() {
		var t Task
		if err := rows.Scan(&t.ID, &t.Name, &t.Project, &t.Models, &t.LocalOnly, &t.Children, &t.ExpiresAt, &t.Revoked, &t.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// RevokeTask revokes one of the user's tasks and every subtask under it.
func RevokeTask(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, userID, taskID string) error {
	if !ids.Valid(taskID) {
		return ErrNotFound
	}
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		var ok bool
		if err := tx.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM tasks WHERE id = $1 AND owner_type = 'user' AND owner_id = $2)", taskID, userID).
			Scan(&ok); err != nil {
			return err
		}
		if !ok {
			return ErrNotFound
		}
		tag, err := tx.Exec(ctx, "UPDATE tasks SET revoked_at = now() WHERE (id = $1 OR $1 = ANY(ancestors)) AND revoked_at IS NULL", taskID)
		if err != nil {
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID = "task.revoked", "task", taskID
		ev.Details = map[string]any{"revoked": tag.RowsAffected()}
		return audit.Append(ctx, tx, ev)
	})
}
