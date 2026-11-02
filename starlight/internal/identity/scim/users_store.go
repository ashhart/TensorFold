package scim

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
)

// userRow is a user as SCIM sees it.
type userRow struct {
	ID               string
	Email            string
	DisplayName      string
	Status           string
	Source           string
	SCIMConnectionID *string
	SCIMUserName     *string
	ExternalID       *string
	Attrs            map[string]any
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

// owned reports whether the connection already owns the user rather than having only signed them in.
func (u userRow) owned() bool { return u.SCIMConnectionID != nil }

// name is the user's SCIM userName, or the email of a user who has only signed in.
func (u userRow) name() string {
	if u.SCIMUserName != nil {
		return *u.SCIMUserName
	}
	return u.Email
}

const userSelect = `SELECT u.id, u.email, u.display_name, u.status, u.source, u.scim_connection_id, u.scim_user_name, u.external_id,
	u.scim_attributes, u.created_at, u.updated_at FROM users u`

// visibleUser admits the connection's own users and just-in-time users who signed in through it; $1 is the connection.
const visibleUser = `u.status <> 'deactivated' AND (u.scim_connection_id = $1 OR (u.scim_connection_id IS NULL AND u.source = 'jit'
	AND EXISTS (SELECT 1 FROM identities i WHERE i.user_id = u.id AND i.connection_id = $1)))`

func scanUserRow(row pgx.Row) (userRow, error) {
	var u userRow
	var raw []byte
	err := row.Scan(&u.ID, &u.Email, &u.DisplayName, &u.Status, &u.Source, &u.SCIMConnectionID, &u.SCIMUserName, &u.ExternalID,
		&raw, &u.CreatedAt, &u.UpdatedAt)
	if err != nil {
		if db.NotFound(err) {
			return u, errNotFound
		}
		return u, err
	}
	u.Attrs = map[string]any{}
	if len(raw) > 0 {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		if err := dec.Decode(&u.Attrs); err != nil {
			return u, err
		}
	}
	return u, nil
}

func loadUser(ctx context.Context, q db.Querier, conn, id string, lock bool) (userRow, error) {
	sql := userSelect + " WHERE u.id = $2 AND " + visibleUser
	if lock {
		sql += " FOR UPDATE OF u"
	}
	return scanUserRow(q.QueryRow(ctx, sql, conn, id))
}

func queryUsers(ctx context.Context, q db.Querier, conn string, lq listQuery) ([]userRow, int, error) {
	where, args := visibleUser, []any{conn}
	if lq.filter != nil {
		cond, fargs, err := compileFilter(resUser, lq.filter, 1)
		if err != nil {
			return nil, 0, err
		}
		where += " AND " + cond
		args = append(args, fargs...)
	}
	var total int
	if err := q.QueryRow(ctx, "SELECT count(*) FROM users u WHERE "+where, args...).Scan(&total); err != nil {
		return nil, 0, err
	}
	if lq.count == 0 || lq.start > total {
		return nil, total, nil
	}
	n := len(args)
	rows, err := q.Query(ctx, userSelect+" WHERE "+where+" ORDER BY u.created_at, u.id LIMIT $"+strconv.Itoa(n+1)+" OFFSET $"+strconv.Itoa(n+2),
		append(args, lq.count, lq.start-1)...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	var out []userRow
	for rows.Next() {
		u, err := scanUserRow(rows)
		if err != nil {
			return nil, 0, err
		}
		out = append(out, u)
	}
	return out, total, rows.Err()
}

// groupRef is a group listed in a user's read-only groups attribute.
type groupRef struct {
	ID          string
	DisplayName string
}

// userGroups loads, in one query, the connection's SCIM groups of each user.
func userGroups(ctx context.Context, q db.Querier, conn string, userIDs []string) (map[string][]groupRef, error) {
	out := map[string][]groupRef{}
	if len(userIDs) == 0 {
		return out, nil
	}
	rows, err := q.Query(ctx, `SELECT gm.user_id, g.id, g.display_name FROM group_members gm JOIN groups g ON g.id = gm.group_id
		WHERE gm.user_id = ANY($2::uuid[]) AND g.source = 'scim' AND g.connection_id = $1 ORDER BY g.display_name, g.id`, conn, userIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var uid string
		var g groupRef
		if err := rows.Scan(&uid, &g.ID, &g.DisplayName); err != nil {
			return nil, err
		}
		out[uid] = append(out[uid], g)
	}
	return out, rows.Err()
}

// userFields are the columns derived from a validated SCIM document.
type userFields struct {
	UserName    string
	Email       string
	DisplayName string
	ExternalID  *string
	Attrs       map[string]any
}

func uniqueness(err error) error {
	switch {
	case db.IsUnique(err, "users_scim_name"):
		return errf(http.StatusConflict, TypeUniqueness, "a user with this userName already exists")
	case db.IsUnique(err, "users_email_live"):
		return errf(http.StatusConflict, TypeUniqueness, "this email already belongs to another account")
	case db.IsUnique(err, ""):
		return errf(http.StatusConflict, TypeUniqueness, "the resource conflicts with an existing one")
	}
	return err
}

// conflictCheck reports a uniqueness error for userName or email before an insert or update.
func conflictCheck(ctx context.Context, q db.Querier, conn, selfID string, f userFields) (adoptable string, err error) {
	var taken bool
	err = q.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM users WHERE scim_connection_id = $1 AND lower(scim_user_name) = lower($2)
		AND status <> 'deactivated' AND id::text <> $3)`, conn, f.UserName, selfID).Scan(&taken)
	if err != nil {
		return "", err
	}
	if taken {
		return "", errf(http.StatusConflict, TypeUniqueness, "a user with this userName already exists")
	}
	var otherID string
	var owner *string
	var source string
	err = q.QueryRow(ctx, "SELECT id, scim_connection_id, source FROM users WHERE email_lower = lower($1) AND status <> 'deactivated' AND id::text <> $2",
		f.Email, selfID).Scan(&otherID, &owner, &source)
	if db.NotFound(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	var signedIn bool
	if owner == nil && source == "jit" {
		if err := q.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM identities WHERE user_id = $1 AND connection_id = $2)", otherID, conn).Scan(&signedIn); err != nil {
			return "", err
		}
	}
	if signedIn && selfID == "" {
		return otherID, nil
	}
	return "", errf(http.StatusConflict, TypeUniqueness, "this email already belongs to another account")
}

// foreignEmail refuses addresses whose domain another connection owns, so one IdP cannot pre-create another's users.
func foreignEmail(ctx context.Context, tx pgx.Tx, conn, email string) error {
	owned, err := mapping.OwnedElsewhere(ctx, tx, conn, email)
	if err != nil {
		return err
	}
	if owned {
		return errf(http.StatusBadRequest, TypeInvalidValue, "this email's domain belongs to another sign-in connection")
	}
	return nil
}

func insertUser(ctx context.Context, tx pgx.Tx, conn string, f userFields, active bool) (string, error) {
	if err := foreignEmail(ctx, tx, conn, f.Email); err != nil {
		return "", err
	}
	status := directory.Active
	if !active {
		status = directory.Suspended
	}
	u, err := directory.CreateUser(ctx, tx, directory.NewUser{Email: f.Email, DisplayName: f.DisplayName, Source: "scim",
		SCIMConnectionID: &conn, SCIMUserName: &f.UserName, ExternalID: f.ExternalID, Status: status})
	if err == directory.ErrConflict {
		return "", errf(http.StatusConflict, TypeUniqueness, "the user conflicts with an existing account")
	}
	if err != nil {
		return "", err
	}
	attrs, _ := json.Marshal(f.Attrs)
	if _, err := tx.Exec(ctx, "UPDATE users SET scim_attributes = $2 WHERE id = $1", u.ID, attrs); err != nil {
		return "", err
	}
	return u.ID, nil
}

// writeUser stores fields on an existing user and takes ownership for the connection when adopting.
func writeUser(ctx context.Context, tx pgx.Tx, conn, id string, f userFields) error {
	if err := foreignEmail(ctx, tx, conn, f.Email); err != nil {
		return err
	}
	attrs, _ := json.Marshal(f.Attrs)
	_, err := tx.Exec(ctx, `UPDATE users SET email = $3, display_name = $4, scim_user_name = $5, external_id = $6, scim_attributes = $7,
		scim_connection_id = $2, source = 'scim', updated_at = now() WHERE id = $1`, id, conn, f.Email, f.DisplayName, f.UserName, f.ExternalID, attrs)
	return uniqueness(err)
}

// setActive moves the user between active and suspended, revoking credentials when suspending.
func setActive(ctx context.Context, tx pgx.Tx, u userRow, active bool) (changed bool, err error) {
	if (u.Status == directory.Active) == active {
		return false, nil
	}
	status, reason := directory.Active, ""
	if !active {
		status, reason = directory.Suspended, "scim_suspended"
	}
	if err := keepOwner(ctx, tx, func() error { return directory.SetStatus(ctx, tx, u.ID, status, reason) }); err != nil {
		if err == directory.ErrConflict {
			return false, errf(http.StatusConflict, TypeUniqueness, "reactivating would duplicate another account's email")
		}
		return false, err
	}
	return true, nil
}

// removeUser deactivates the user, drops the connection's memberships and identity links, and revokes everything.
func removeUser(ctx context.Context, tx pgx.Tx, conn string, u userRow) error {
	if err := keepOwner(ctx, tx, func() error { return directory.SetStatus(ctx, tx, u.ID, directory.Deactivated, "scim_deleted") }); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `DELETE FROM group_members gm USING groups g WHERE gm.group_id = g.id AND gm.user_id = $1
		AND g.source = 'scim' AND g.connection_id = $2`, u.ID, conn); err != nil {
		return err
	}
	_, err := tx.Exec(ctx, "DELETE FROM identities WHERE user_id = $1 AND connection_id = $2", u.ID, conn)
	return err
}
