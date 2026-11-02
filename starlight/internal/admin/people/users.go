package people

import (
	"context"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// UserRow is one line of the user list.
type UserRow struct {
	ID          string
	Email       string
	DisplayName string
	Status      string
	Source      string
	LastLoginAt *time.Time
	CreatedAt   time.Time
}

// Filter narrows the user list; Query matches email or name.
type Filter struct {
	Query  string
	Status string
	Offset int
	Limit  int
}

// ListUsers returns one page of users and the total that match.
func ListUsers(ctx context.Context, q db.Querier, f Filter) ([]UserRow, int, error) {
	if f.Limit <= 0 || f.Limit > 200 {
		f.Limit = 50
	}
	f.Offset = max(f.Offset, 0)
	like := "%" + strings.NewReplacer(`\`, `\\`, "%", `\%`, "_", `\_`).Replace(strings.ToLower(strings.TrimSpace(f.Query))) + "%"
	where := `WHERE ($1 = '%%' OR email_lower LIKE $1 OR lower(display_name) LIKE $1) AND ($2 = '' OR status = $2)`
	var total int
	if err := q.QueryRow(ctx, "SELECT count(*) FROM users "+where, like, f.Status).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := q.Query(ctx, `SELECT id, email, display_name, status, source, last_login_at, created_at FROM users `+where+`
		ORDER BY email_lower LIMIT $3 OFFSET $4`, like, f.Status, f.Limit, f.Offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	var out []UserRow
	for rows.Next() {
		var u UserRow
		if err := rows.Scan(&u.ID, &u.Email, &u.DisplayName, &u.Status, &u.Source, &u.LastLoginAt, &u.CreatedAt); err != nil {
			return nil, 0, err
		}
		out = append(out, u)
	}
	return out, total, rows.Err()
}

// Identity is one sign-in identity linked to a user.
type Identity struct {
	Connection  string
	Slug        string
	Subject     string
	Email       string
	LastLoginAt *time.Time
}

// UserDetail is everything the user page shows.
type UserDetail struct {
	User         directory.User
	Owner        bool
	Identities   []Identity
	Groups       []directory.Group
	Bindings     []Binding
	Sessions     int
	Devices      int
	AccessTokens int
	Keys         []keys.Key
	Tasks        []Task
}

// GetUser loads a user with their identities, groups, bindings, credentials and tasks.
func GetUser(ctx context.Context, q db.Querier, id string) (UserDetail, error) {
	var d UserDetail
	u, err := directory.GetUser(ctx, q, id)
	if err != nil {
		return d, err
	}
	d.User = u
	if d.Owner, err = IsOwner(ctx, q, id); err != nil {
		return d, err
	}
	rows, err := q.Query(ctx, `SELECT c.name, c.slug, i.subject, i.email, i.last_login_at FROM identities i JOIN idp_connections c ON c.id = i.connection_id
		WHERE i.user_id = $1 ORDER BY c.name`, id)
	if err != nil {
		return d, err
	}
	for rows.Next() {
		var i Identity
		if err := rows.Scan(&i.Connection, &i.Slug, &i.Subject, &i.Email, &i.LastLoginAt); err != nil {
			rows.Close()
			return d, err
		}
		d.Identities = append(d.Identities, i)
	}
	rows.Close()
	if d.Groups, err = directory.UserGroups(ctx, q, id); err != nil {
		return d, err
	}
	if d.Bindings, err = UserBindings(ctx, q, id); err != nil {
		return d, err
	}
	err = q.QueryRow(ctx, `SELECT
		(SELECT count(*) FROM sessions WHERE user_id = $1 AND revoked_at IS NULL AND expires_at > now() AND idle_expires_at > now()),
		(SELECT count(DISTINCT family_id) FROM refresh_tokens WHERE user_id = $1 AND revoked_at IS NULL AND used_at IS NULL
			AND idle_expires_at > now() AND family_expires_at > now()),
		(SELECT count(*) FROM access_tokens WHERE user_id = $1 AND revoked_at IS NULL AND expires_at > now())`, id).
		Scan(&d.Sessions, &d.Devices, &d.AccessTokens)
	if err != nil {
		return d, err
	}
	if d.Keys, err = keys.List(ctx, q, keys.Filter{OwnerType: "user", OwnerID: id}); err != nil {
		return d, err
	}
	d.Tasks, err = Tasks(ctx, q, id)
	return d, err
}

// SetStatus suspends, reactivates or deactivates a user; leaving active revokes every credential in one transaction.
func SetStatus(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, userID, status string) error {
	if !ids.Valid(userID) {
		return ErrNotFound
	}
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		check, err := GuardOwners(ctx, tx)
		if err != nil {
			return err
		}
		before, err := directory.GetUser(ctx, tx, userID)
		if err != nil {
			return err
		}
		if err := directory.SetStatus(ctx, tx, userID, status, "admin:"+status); err != nil {
			return err
		}
		if err := check(); err != nil {
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID = "user.status", "user", userID
		ev.Details = map[string]any{"email": before.Email, "from": before.Status, "to": status}
		return audit.Append(ctx, tx, ev)
	})
}

// RevokeAll ends every session, token, key and task the user holds without changing their status.
func RevokeAll(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, userID string) error {
	if !ids.Valid(userID) {
		return ErrNotFound
	}
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		u, err := directory.GetUser(ctx, tx, userID)
		if err != nil {
			return err
		}
		if err := directory.RevokeUser(ctx, tx, userID, "admin:revoke"); err != nil {
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID = "user.revoked", "user", userID
		ev.Details = map[string]any{"email": u.Email}
		return audit.Append(ctx, tx, ev)
	})
}
