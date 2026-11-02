package directory

import (
	"context"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Group sources.
const (
	FromClaim = "claim"
	FromSCIM  = "scim"
	FromLocal = "local"
)

// Group is a set of users that role bindings and project membership can name.
type Group struct {
	ID           string
	ConnectionID *string
	Source       string
	ExternalID   string
	DisplayName  string
}

// GroupRef identifies a group as an identity provider reports it.
type GroupRef struct {
	ExternalID  string
	DisplayName string
}

// UpsertGroup returns the id of the group, creating or renaming it as needed.
func UpsertGroup(ctx context.Context, q db.Querier, connectionID *string, source string, ref GroupRef) (string, error) {
	name := strings.TrimSpace(ref.DisplayName)
	if name == "" {
		name = ref.ExternalID
	}
	var id string
	err := q.QueryRow(ctx, `INSERT INTO groups (id, connection_id, source, external_id, display_name) VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (connection_id, source, external_id) DO UPDATE SET display_name = EXCLUDED.display_name, updated_at = now()
		RETURNING id`, ids.New(), connectionID, source, ref.ExternalID, name).Scan(&id)
	return id, err
}

// SetClaimGroups makes the user's claim-sourced memberships at one connection exactly refs.
func SetClaimGroups(ctx context.Context, q db.Querier, userID, connectionID string, refs []GroupRef) error {
	keep := make([]string, 0, len(refs))
	for _, r := range refs {
		if r.ExternalID == "" {
			continue
		}
		id, err := UpsertGroup(ctx, q, &connectionID, FromClaim, r)
		if err != nil {
			return err
		}
		keep = append(keep, id)
		if _, err := q.Exec(ctx, `INSERT INTO group_members (group_id, user_id) VALUES ($1, $2)
			ON CONFLICT (group_id, user_id) DO UPDATE SET updated_at = now()`, id, userID); err != nil {
			return err
		}
	}
	_, err := q.Exec(ctx, `DELETE FROM group_members gm USING groups g
		WHERE gm.group_id = g.id AND gm.user_id = $1 AND g.connection_id = $2 AND g.source = 'claim' AND NOT (g.id = ANY($3::uuid[]))`,
		userID, connectionID, keep)
	return err
}

// UserGroups lists every group the user belongs to.
func UserGroups(ctx context.Context, q db.Querier, userID string) ([]Group, error) {
	rows, err := q.Query(ctx, `SELECT g.id, g.connection_id, g.source, g.external_id, g.display_name
		FROM groups g JOIN group_members gm ON gm.group_id = g.id WHERE gm.user_id = $1 ORDER BY g.display_name`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Group
	for rows.Next() {
		var g Group
		if err := rows.Scan(&g.ID, &g.ConnectionID, &g.Source, &g.ExternalID, &g.DisplayName); err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, rows.Err()
}

// GroupIDs returns just the ids from UserGroups.
func GroupIDs(ctx context.Context, q db.Querier, userID string) ([]string, error) {
	gs, err := UserGroups(ctx, q, userID)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(gs))
	for i, g := range gs {
		out[i] = g.ID
	}
	return out, nil
}
