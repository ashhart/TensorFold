package people

import (
	"context"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// GroupRow is one line of the group list.
type GroupRow struct {
	ID          string
	DisplayName string
	Source      string
	ExternalID  string
	Connection  string
	Members     int
	Bindings    int
}

const groupCols = `g.id, g.display_name, g.source, g.external_id, coalesce(c.name, ''),
	(SELECT count(*) FROM group_members gm WHERE gm.group_id = g.id),
	(SELECT count(*) FROM role_bindings rb WHERE rb.subject_type = 'group' AND rb.subject_id = g.id)`

// ListGroups returns groups whose name or external id matches query, most populated first.
func ListGroups(ctx context.Context, q db.Querier, query string) ([]GroupRow, error) {
	like := "%" + strings.NewReplacer(`\`, `\\`, "%", `\%`, "_", `\_`).Replace(strings.ToLower(strings.TrimSpace(query))) + "%"
	rows, err := q.Query(ctx, `SELECT `+groupCols+` FROM groups g LEFT JOIN idp_connections c ON c.id = g.connection_id
		WHERE lower(g.display_name) LIKE $1 OR lower(g.external_id) LIKE $1 ORDER BY 6 DESC, g.display_name LIMIT 500`, like)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []GroupRow
	for rows.Next() {
		var g GroupRow
		if err := rows.Scan(&g.ID, &g.DisplayName, &g.Source, &g.ExternalID, &g.Connection, &g.Members, &g.Bindings); err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, rows.Err()
}

// GroupDetail is a group with its members and role bindings.
type GroupDetail struct {
	Group    GroupRow
	Members  []UserRow
	Bindings []Binding
}

// GetGroup loads a group, up to 500 of its members and its bindings.
func GetGroup(ctx context.Context, q db.Querier, id string) (GroupDetail, error) {
	var d GroupDetail
	if !ids.Valid(id) {
		return d, ErrNotFound
	}
	err := q.QueryRow(ctx, `SELECT `+groupCols+` FROM groups g LEFT JOIN idp_connections c ON c.id = g.connection_id WHERE g.id = $1`, id).
		Scan(&d.Group.ID, &d.Group.DisplayName, &d.Group.Source, &d.Group.ExternalID, &d.Group.Connection, &d.Group.Members, &d.Group.Bindings)
	if db.NotFound(err) {
		return d, ErrNotFound
	}
	if err != nil {
		return d, err
	}
	rows, err := q.Query(ctx, `SELECT u.id, u.email, u.display_name, u.status, u.source, u.last_login_at, u.created_at FROM users u
		JOIN group_members gm ON gm.user_id = u.id WHERE gm.group_id = $1 ORDER BY u.email_lower LIMIT 500`, id)
	if err != nil {
		return d, err
	}
	for rows.Next() {
		var u UserRow
		if err := rows.Scan(&u.ID, &u.Email, &u.DisplayName, &u.Status, &u.Source, &u.LastLoginAt, &u.CreatedAt); err != nil {
			rows.Close()
			return d, err
		}
		d.Members = append(d.Members, u)
	}
	rows.Close()
	d.Bindings, err = SubjectBindings(ctx, q, "group", id)
	return d, err
}
