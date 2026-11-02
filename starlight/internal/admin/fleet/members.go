package fleet

import (
	"context"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Binding is one role held by a user, group or service account, in a project or organisation-wide.
type Binding struct {
	ID          string
	Role        string
	SubjectType string
	SubjectID   string
	Label       string
	Detail      string
	ProjectID   string
}

// Bindings lists role bindings in a project, or organisation-wide ones when projectID is empty.
func Bindings(ctx context.Context, q db.Querier, projectID string) ([]Binding, error) {
	rows, err := q.Query(ctx, `SELECT rb.id, rb.role, rb.subject_type, rb.subject_id, coalesce(rb.project_id::text, ''),
		coalesce(u.email, g.display_name, sa.name, rb.subject_id::text),
		coalesce(CASE WHEN rb.subject_type = 'group' THEN coalesce(c.slug, 'local') || ' · ' || g.external_id END,
			CASE WHEN rb.subject_type = 'user' THEN u.status END, '')
		FROM role_bindings rb
		LEFT JOIN users u ON rb.subject_type = 'user' AND u.id = rb.subject_id
		LEFT JOIN groups g ON rb.subject_type = 'group' AND g.id = rb.subject_id
		LEFT JOIN idp_connections c ON c.id = g.connection_id
		LEFT JOIN service_accounts sa ON rb.subject_type = 'service_account' AND sa.id = rb.subject_id
		WHERE rb.project_id IS NOT DISTINCT FROM $1 ORDER BY rb.role, 6`, optID(projectID))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Binding
	for rows.Next() {
		var b Binding
		if err := rows.Scan(&b.ID, &b.Role, &b.SubjectType, &b.SubjectID, &b.ProjectID, &b.Label, &b.Detail); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, rows.Err()
}

// Bind grants role to a subject; projectID empty means organisation scope.
func (t *Tx) Bind(ctx context.Context, role, subjectType, subjectID, projectID string) (bool, error) {
	if !authz.ValidRole(role, projectID != "") {
		if projectID != "" {
			return false, Invalid("project roles are project_admin, member and viewer")
		}
		return false, Invalid("unknown organisation role " + role)
	}
	if subjectType != "user" && subjectType != "group" && subjectType != "service_account" || !ids.Valid(subjectID) {
		return false, Invalid("choose a user, group or service account")
	}
	tag, err := t.tx.Exec(ctx, `INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id, created_by) VALUES ($1,$2,$3,$4,$5,$6)
		ON CONFLICT (role, subject_type, subject_id, project_id) DO NOTHING`, ids.New(), role, subjectType, subjectID, optID(projectID),
		optID(t.actor.ActorID))
	if err != nil || tag.RowsAffected() == 0 {
		return false, err
	}
	return true, t.record(ctx, "role.bound", subjectType, subjectID, map[string]any{"role": role, "project_id": projectID})
}

// Unbind removes one binding within the given scope.
func (t *Tx) Unbind(ctx context.Context, bindingID, projectID string) error {
	keepOwner, err := people.GuardOwners(ctx, t.tx)
	if err != nil {
		return err
	}
	var role, st, sid string
	err = t.tx.QueryRow(ctx, `DELETE FROM role_bindings WHERE id = $1 AND project_id IS NOT DISTINCT FROM $2 RETURNING role, subject_type, subject_id`,
		bindingID, optID(projectID)).Scan(&role, &st, &sid)
	if err != nil {
		return notFound(err)
	}
	if err := keepOwner(); err != nil {
		return Invalid("that would leave the organisation without an active owner; make someone else an owner first")
	}
	return t.record(ctx, "role.unbound", st, sid, map[string]any{"role": role, "project_id": projectID})
}

// UserByEmail finds a live user, who must already exist through sign-in or SCIM.
func UserByEmail(ctx context.Context, q db.Querier, email string) (directory.User, error) {
	u, err := directory.FindByEmail(ctx, q, strings.TrimSpace(email))
	if err == directory.ErrNotFound {
		return u, Invalid("no user with email " + email + "; people appear after their first sign-in or SCIM provisioning")
	}
	return u, err
}

// GroupChoice is a group offered in member pickers.
type GroupChoice struct {
	ID         string
	Label      string
	Connection string
	ExternalID string
}

// GroupChoices lists every group with the connection it came from.
func GroupChoices(ctx context.Context, q db.Querier) ([]GroupChoice, error) {
	rows, err := q.Query(ctx, `SELECT g.id, g.display_name, coalesce(c.slug, ''), g.external_id FROM groups g
		LEFT JOIN idp_connections c ON c.id = g.connection_id ORDER BY g.display_name, c.slug`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []GroupChoice
	for rows.Next() {
		var g GroupChoice
		if err := rows.Scan(&g.ID, &g.Label, &g.Connection, &g.ExternalID); err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, rows.Err()
}

// GroupsFor finds the groups a connection reports as externalID, creating a claim group when there is none.
func (t *Tx) GroupsFor(ctx context.Context, connectionSlug, externalID string) (found []string, created bool, err error) {
	var connID string
	if err := t.tx.QueryRow(ctx, "SELECT id FROM idp_connections WHERE slug = $1", connectionSlug).Scan(&connID); err != nil {
		if db.NotFound(err) {
			return nil, false, Invalid("no identity provider connection named " + connectionSlug)
		}
		return nil, false, err
	}
	rows, err := t.tx.Query(ctx, "SELECT id FROM groups WHERE connection_id = $1 AND external_id = $2 ORDER BY source", connID, externalID)
	if err != nil {
		return nil, false, err
	}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return nil, false, err
		}
		found = append(found, id)
	}
	rows.Close()
	if len(found) > 0 || rows.Err() != nil {
		return found, false, rows.Err()
	}
	id, err := directory.UpsertGroup(ctx, t.tx, &connID, directory.FromClaim, directory.GroupRef{ExternalID: externalID})
	if err != nil {
		return nil, false, err
	}
	return []string{id}, true, t.record(ctx, "group.declared", "group", id, map[string]any{"connection": connectionSlug, "external_id": externalID})
}
