package authz

import (
	"context"
	"sort"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Grants are a principal's effective roles, loaded once per request.
type Grants struct {
	Org      map[string]bool
	Projects map[string]map[string]bool
}

// Can reports whether the grants allow p; projectID is ignored for organisation permissions.
func (g Grants) Can(p Perm, projectID string) bool {
	for name := range g.Org {
		r := roles[name]
		if has(r.org, p) || projectID != "" && has(r.project, p) {
			return true
		}
	}
	if projectID == "" {
		return false
	}
	for name := range g.Projects[projectID] {
		if has(roles[name].project, p) {
			return true
		}
	}
	return false
}

// EveryProject reports whether an organisation role grants p in every project, such as an owner approving any plan.
func (g Grants) EveryProject(p Perm) bool {
	for name := range g.Org {
		if has(roles[name].project, p) {
			return true
		}
	}
	return false
}

// AnyOrg reports whether any organisation role is held, which opens the admin console.
func (g Grants) AnyOrg() bool { return len(g.Org) > 0 }

// ProjectIDs lists projects where p is granted directly, sorted.
func (g Grants) ProjectIDs(p Perm) []string {
	var out []string
	for id, rs := range g.Projects {
		for name := range rs {
			if has(roles[name].project, p) {
				out = append(out, id)
				break
			}
		}
	}
	sort.Strings(out)
	return out
}

// TopRole names the strongest project role held, for display.
func (g Grants) TopRole(projectID string) string {
	for _, n := range []string{"project_admin", "member", "viewer"} {
		if g.Projects[projectID][n] {
			return n
		}
	}
	return ""
}

// LoadUser resolves bindings for the user and their groups, plus membership of everyone-projects.
func LoadUser(ctx context.Context, q db.Querier, userID string, groupIDs []string) (Grants, error) {
	if groupIDs == nil {
		groupIDs = []string{}
	}
	rows, err := q.Query(ctx, `SELECT rb.role, rb.project_id FROM role_bindings rb
		LEFT JOIN projects p ON p.id = rb.project_id
		WHERE ((rb.subject_type = 'user' AND rb.subject_id = $1) OR (rb.subject_type = 'group' AND rb.subject_id = ANY($2::uuid[])))
		AND (rb.project_id IS NULL OR p.archived_at IS NULL)
		UNION SELECT 'member', id FROM projects WHERE everyone AND archived_at IS NULL`, userID, groupIDs)
	if err != nil {
		return Grants{}, err
	}
	return collect(rows)
}

// LoadServiceAccount gives a service account member rights in its own project plus any explicit bindings.
func LoadServiceAccount(ctx context.Context, q db.Querier, id string) (Grants, error) {
	rows, err := q.Query(ctx, `SELECT rb.role, rb.project_id FROM role_bindings rb
		WHERE rb.subject_type = 'service_account' AND rb.subject_id = $1 AND rb.project_id IS NOT NULL
		UNION SELECT 'member', sa.project_id FROM service_accounts sa JOIN projects p ON p.id = sa.project_id
		WHERE sa.id = $1 AND sa.status = 'active' AND p.archived_at IS NULL`, id)
	if err != nil {
		return Grants{}, err
	}
	return collect(rows)
}

type rowScanner interface {
	Next() bool
	Scan(...any) error
	Close()
	Err() error
}

func collect(rows rowScanner) (Grants, error) {
	defer rows.Close()
	g := Grants{Org: map[string]bool{}, Projects: map[string]map[string]bool{}}
	for rows.Next() {
		var role string
		var project *string
		if err := rows.Scan(&role, &project); err != nil {
			return g, err
		}
		if _, ok := roles[role]; !ok {
			continue
		}
		if project == nil {
			if !roles[role].scoped {
				g.Org[role] = true
			}
			continue
		}
		if roles[role].scoped {
			if g.Projects[*project] == nil {
				g.Projects[*project] = map[string]bool{}
			}
			g.Projects[*project][role] = true
		}
	}
	return g, rows.Err()
}
