package scim

import (
	"context"
	"net/http"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// groupRow is a SCIM group; ExternalID equals ID when the IdP sent no externalId.
type groupRow struct {
	ID          string
	ExternalID  string
	DisplayName string
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

const groupSelect = `SELECT g.id, g.external_id, g.display_name, g.created_at, g.updated_at FROM groups g`

// visibleGroup admits only groups this connection provisioned; $1 is the connection.
const visibleGroup = `g.source = 'scim' AND g.connection_id = $1`

func scanGroup(row pgx.Row) (groupRow, error) {
	var g groupRow
	err := row.Scan(&g.ID, &g.ExternalID, &g.DisplayName, &g.CreatedAt, &g.UpdatedAt)
	if db.NotFound(err) {
		return g, errNotFound
	}
	return g, err
}

func loadGroup(ctx context.Context, q db.Querier, conn, id string, lock bool) (groupRow, error) {
	sql := groupSelect + " WHERE g.id = $2 AND " + visibleGroup
	if lock {
		sql += " FOR UPDATE"
	}
	return scanGroup(q.QueryRow(ctx, sql, conn, id))
}

func queryGroups(ctx context.Context, q db.Querier, conn string, lq listQuery) ([]groupRow, int, error) {
	where, args := visibleGroup, []any{conn}
	if lq.filter != nil {
		cond, fargs, err := compileFilter(resGroup, lq.filter, 1)
		if err != nil {
			return nil, 0, err
		}
		where += " AND " + cond
		args = append(args, fargs...)
	}
	var total int
	if err := q.QueryRow(ctx, "SELECT count(*) FROM groups g WHERE "+where, args...).Scan(&total); err != nil {
		return nil, 0, err
	}
	if lq.count == 0 || lq.start > total {
		return nil, total, nil
	}
	n := len(args)
	rows, err := q.Query(ctx, groupSelect+" WHERE "+where+" ORDER BY g.created_at, g.id LIMIT $"+strconv.Itoa(n+1)+" OFFSET $"+strconv.Itoa(n+2),
		append(args, lq.count, lq.start-1)...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	var out []groupRow
	for rows.Next() {
		g, err := scanGroup(rows)
		if err != nil {
			return nil, 0, err
		}
		out = append(out, g)
	}
	return out, total, rows.Err()
}

// member is one live user in a group.
type member struct {
	ID    string
	Label string
}

func groupMembers(ctx context.Context, q db.Querier, groupIDs []string) (map[string][]member, error) {
	out := map[string][]member{}
	if len(groupIDs) == 0 {
		return out, nil
	}
	rows, err := q.Query(ctx, `SELECT gm.group_id, u.id, coalesce(nullif(u.display_name, ''), u.email) FROM group_members gm
		JOIN users u ON u.id = gm.user_id WHERE gm.group_id = ANY($1::uuid[]) AND u.status <> 'deactivated' ORDER BY u.created_at, u.id`, groupIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var gid string
		var m member
		if err := rows.Scan(&gid, &m.ID, &m.Label); err != nil {
			return nil, err
		}
		out[gid] = append(out[gid], m)
	}
	return out, rows.Err()
}

func groupUniqueness(err error) error {
	switch {
	case db.IsUnique(err, "groups_scim_name"):
		return errf(http.StatusConflict, TypeUniqueness, "a group with this displayName already exists")
	case db.IsUnique(err, ""):
		return errf(http.StatusConflict, TypeUniqueness, "a group with this externalId already exists")
	}
	return err
}

func insertGroup(ctx context.Context, tx pgx.Tx, conn, name string, ext *string) (string, error) {
	id := ids.New()
	external := id
	if ext != nil {
		external = *ext
	}
	_, err := tx.Exec(ctx, "INSERT INTO groups (id, connection_id, source, external_id, display_name) VALUES ($1, $2, 'scim', $3, $4)",
		id, conn, external, name)
	return id, groupUniqueness(err)
}

func renameGroup(ctx context.Context, tx pgx.Tx, id, name string, ext *string) error {
	external := id
	if ext != nil {
		external = *ext
	}
	_, err := tx.Exec(ctx, "UPDATE groups SET display_name = $2, external_id = $3, updated_at = now() WHERE id = $1", id, name, external)
	return groupUniqueness(err)
}

// checkMembers refuses member ids that are not live users this connection can see.
func checkMembers(ctx context.Context, q db.Querier, conn string, userIDs []string) error {
	if len(userIDs) == 0 {
		return nil
	}
	rows, err := q.Query(ctx, "SELECT u.id FROM users u WHERE u.id = ANY($2::uuid[]) AND "+visibleUser, conn, userIDs)
	if err != nil {
		return err
	}
	defer rows.Close()
	found := map[string]bool{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return err
		}
		found[id] = true
	}
	if err := rows.Err(); err != nil {
		return err
	}
	for _, id := range userIDs {
		if !found[id] {
			return badRequest(TypeInvalidValue, "member %s is not a user provisioned through this connection", id)
		}
	}
	return nil
}

// syncMembers makes the group's membership exactly want and returns what changed.
func syncMembers(ctx context.Context, tx pgx.Tx, groupID string, current []member, want []string) (added, removed []string, err error) {
	have := map[string]bool{}
	for _, m := range current {
		have[m.ID] = true
	}
	keep := map[string]bool{}
	for _, id := range want {
		keep[id] = true
		if !have[id] {
			added = append(added, id)
		}
	}
	for _, m := range current {
		if !keep[m.ID] {
			removed = append(removed, m.ID)
		}
	}
	if len(added) > 0 {
		if _, err := tx.Exec(ctx, `INSERT INTO group_members (group_id, user_id) SELECT $1, unnest($2::uuid[])
			ON CONFLICT (group_id, user_id) DO UPDATE SET updated_at = now()`, groupID, added); err != nil {
			return nil, nil, err
		}
	}
	if len(removed) > 0 {
		if _, err := tx.Exec(ctx, "DELETE FROM group_members WHERE group_id = $1 AND user_id = ANY($2::uuid[])", groupID, removed); err != nil {
			return nil, nil, err
		}
	}
	if len(added)+len(removed) > 0 {
		_, err = tx.Exec(ctx, "UPDATE groups SET updated_at = now() WHERE id = $1", groupID)
	}
	return added, removed, err
}

// dropGroup deletes the group and every role binding that named it, leaving no dangling grants.
func dropGroup(ctx context.Context, tx pgx.Tx, id string) error {
	if _, err := tx.Exec(ctx, "DELETE FROM role_bindings WHERE subject_type = 'group' AND subject_id = $1", id); err != nil {
		return err
	}
	_, err := tx.Exec(ctx, "DELETE FROM groups WHERE id = $1", id)
	return err
}
