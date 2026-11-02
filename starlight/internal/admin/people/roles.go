package people

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Binding is one role granted to a user, group or service account at organisation or project scope.
type Binding struct {
	ID           string
	Role         string
	SubjectType  string
	SubjectID    string
	SubjectLabel string
	ProjectID    *string
	ProjectSlug  string
	ProjectName  string
	Via          string
	CreatedAt    time.Time
}

// Org reports whether the binding is at organisation scope.
func (b Binding) Org() bool { return b.ProjectID == nil }

// ErrExists means the subject already holds that role at that scope.
var ErrExists = errors.New("that role is already granted")

// ErrRole means the role does not exist at the requested scope.
var ErrRole = errors.New("that role cannot be granted at this scope")

const bindingCols = `rb.id, rb.role, rb.subject_type, rb.subject_id,
	coalesce(u.email, g.display_name, sa.name, ''), rb.project_id, coalesce(p.slug, ''), coalesce(p.name, ''), rb.created_at`

const bindingJoins = ` FROM role_bindings rb
	LEFT JOIN users u ON rb.subject_type = 'user' AND u.id = rb.subject_id
	LEFT JOIN groups g ON rb.subject_type = 'group' AND g.id = rb.subject_id
	LEFT JOIN service_accounts sa ON rb.subject_type = 'service_account' AND sa.id = rb.subject_id
	LEFT JOIN projects p ON p.id = rb.project_id`

func scanBindings(rows pgx.Rows) ([]Binding, error) {
	defer rows.Close()
	var out []Binding
	for rows.Next() {
		var b Binding
		if err := rows.Scan(&b.ID, &b.Role, &b.SubjectType, &b.SubjectID, &b.SubjectLabel, &b.ProjectID, &b.ProjectSlug, &b.ProjectName,
			&b.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, rows.Err()
}

// OrgBindings lists every organisation-scope role binding, owners first.
func OrgBindings(ctx context.Context, q db.Querier) ([]Binding, error) {
	rows, err := q.Query(ctx, "SELECT "+bindingCols+bindingJoins+` WHERE rb.project_id IS NULL
		ORDER BY rb.role <> 'owner', rb.role, 5`)
	if err != nil {
		return nil, err
	}
	return scanBindings(rows)
}

// SubjectBindings lists the bindings held directly by one user, group or service account.
func SubjectBindings(ctx context.Context, q db.Querier, subjectType, subjectID string) ([]Binding, error) {
	rows, err := q.Query(ctx, "SELECT "+bindingCols+bindingJoins+` WHERE rb.subject_type = $1 AND rb.subject_id = $2
		ORDER BY rb.project_id NULLS FIRST, p.slug, rb.role`, subjectType, subjectID)
	if err != nil {
		return nil, err
	}
	return scanBindings(rows)
}

// UserBindings lists a user's direct bindings and those inherited from groups, naming the group in Via.
func UserBindings(ctx context.Context, q db.Querier, userID string) ([]Binding, error) {
	rows, err := q.Query(ctx, "SELECT "+bindingCols+`, coalesce(gv.display_name, '')`+bindingJoins+`
		LEFT JOIN group_members gm ON rb.subject_type = 'group' AND gm.group_id = rb.subject_id AND gm.user_id = $1
		LEFT JOIN groups gv ON gv.id = gm.group_id
		WHERE (rb.subject_type = 'user' AND rb.subject_id = $1) OR gm.user_id IS NOT NULL
		ORDER BY rb.project_id NULLS FIRST, p.slug, rb.role`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Binding
	for rows.Next() {
		var b Binding
		if err := rows.Scan(&b.ID, &b.Role, &b.SubjectType, &b.SubjectID, &b.SubjectLabel, &b.ProjectID, &b.ProjectSlug, &b.ProjectName,
			&b.CreatedAt, &b.Via); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, rows.Err()
}

// Grant gives a subject a role at organisation scope, or at project scope when projectID is set.
func Grant(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, role, subjectType, subjectID, projectID string) (string, error) {
	if !authz.ValidRole(role, projectID != "") {
		return "", ErrRole
	}
	if subjectType != "user" && subjectType != "group" || !ids.Valid(subjectID) || projectID != "" && !ids.Valid(projectID) {
		return "", ErrNotFound
	}
	id := ids.New()
	var project, by any
	if projectID != "" {
		project = projectID
	}
	if ids.Valid(ev.ActorID) {
		by = ev.ActorID
	}
	err := db.Tx(ctx, pool, func(tx pgx.Tx) error {
		var label string
		var err error
		if subjectType == "user" {
			err = tx.QueryRow(ctx, "SELECT email FROM users WHERE id = $1 AND status <> 'deactivated'", subjectID).Scan(&label)
		} else {
			err = tx.QueryRow(ctx, "SELECT display_name FROM groups WHERE id = $1", subjectID).Scan(&label)
		}
		if db.NotFound(err) {
			return ErrNotFound
		}
		if err != nil {
			return err
		}
		if projectID != "" {
			var ok bool
			if err := tx.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM projects WHERE id = $1 AND archived_at IS NULL)", projectID).Scan(&ok); err != nil {
				return err
			}
			if !ok {
				return ErrNotFound
			}
		}
		tag, err := tx.Exec(ctx, `INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id, created_by)
			VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING`, id, role, subjectType, subjectID, project, by)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return ErrExists
		}
		ev.Action, ev.TargetType, ev.TargetID = "role.granted", subjectType, subjectID
		ev.Details = map[string]any{"role": role, "subject": label, "project_id": projectID, "binding_id": id}
		return audit.Append(ctx, tx, ev)
	})
	return id, err
}

// GetBinding loads one binding.
func GetBinding(ctx context.Context, q db.Querier, id string) (Binding, error) {
	if !ids.Valid(id) {
		return Binding{}, ErrNotFound
	}
	rows, err := q.Query(ctx, "SELECT "+bindingCols+bindingJoins+" WHERE rb.id = $1", id)
	if err != nil {
		return Binding{}, err
	}
	bs, err := scanBindings(rows)
	if err != nil {
		return Binding{}, err
	}
	if len(bs) == 0 {
		return Binding{}, ErrNotFound
	}
	return bs[0], nil
}

// Revoke removes a binding, refusing to remove the last route to an active owner.
func Revoke(ctx context.Context, pool *pgxpool.Pool, ev audit.Event, bindingID string) error {
	b, err := GetBinding(ctx, pool, bindingID)
	if err != nil {
		return err
	}
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		check, err := GuardOwners(ctx, tx)
		if err != nil {
			return err
		}
		tag, err := tx.Exec(ctx, "DELETE FROM role_bindings WHERE id = $1", bindingID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return ErrNotFound
		}
		if err := check(); err != nil {
			return err
		}
		ev.Action, ev.TargetType, ev.TargetID = "role.revoked", b.SubjectType, b.SubjectID
		project := ""
		if b.ProjectID != nil {
			project = *b.ProjectID
		}
		ev.Details = map[string]any{"role": b.Role, "subject": b.SubjectLabel, "project_id": project, "binding_id": b.ID}
		return audit.Append(ctx, tx, ev)
	})
}

// ProjectRef names a project in pickers.
type ProjectRef struct {
	ID   string
	Slug string
	Name string
}

// Projects lists live projects for role pickers.
func Projects(ctx context.Context, q db.Querier) ([]ProjectRef, error) {
	rows, err := q.Query(ctx, "SELECT id, slug, name FROM projects WHERE archived_at IS NULL ORDER BY slug")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ProjectRef
	for rows.Next() {
		var p ProjectRef
		if err := rows.Scan(&p.ID, &p.Slug, &p.Name); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}
