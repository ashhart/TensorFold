package plans

import (
	"context"
	"net/http"

	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/principal"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

func (s *Service) get(w http.ResponseWriter, r *http.Request) {
	c, err := s.caller(r)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	p, err := s.view(r.Context(), c, r.PathValue("id"))
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	httpx.JSON(w, http.StatusOK, p)
}

// view shows a plan to its owner or an approver; the owner receives the task token once, after approval.
func (s *Service) view(ctx context.Context, c *authn.Caller, id string) (Plan, error) {
	var p Plan
	var ownerType, ownerID string
	var sealed, planSum, egress, token []byte
	if !validID(id) {
		return p, httpx.ErrNotFound
	}
	err := s.Pool.QueryRow(ctx, `SELECT id, project_id, owner_type, owner_id, title, planner_model, executor_model, status, plan_sealed, plan_sha256,
		egress_sha256, egress_bytes, task_id, task_token_sealed, error FROM plans WHERE id = $1`, id).
		Scan(&p.ID, &p.ProjectID, &ownerType, &ownerID, &p.Title, &p.Planner, &p.Executor, &p.Status, &sealed, &planSum, &egress, &p.EgressBytes,
			&p.TaskID, &token, &p.Error)
	if err != nil {
		return p, httpx.ErrNotFound
	}
	owner := c.Task == nil && ownerType == string(c.P.Type) && ownerID == c.P.ID
	if !owner && !canApprove(c, p.ProjectID) {
		return Plan{}, httpx.ErrNotFound
	}
	p.EgressSHA256 = hexOf(egress)
	if len(planSum) > 0 {
		p.PlanSHA256 = hexOf(planSum)
	}
	if len(sealed) > 0 {
		text, err := s.Keys.Open(sealed, "plans.plan:"+id)
		if err != nil {
			return p, err
		}
		p.Text = string(text)
	}
	if owner && len(token) > 0 {
		tag, err := s.Pool.Exec(ctx, "UPDATE plans SET task_token_sealed = NULL WHERE id = $1 AND task_token_sealed IS NOT NULL", id)
		if err == nil && tag.RowsAffected() == 1 {
			if tok, err := s.Keys.Open(token, "plans.task_token:"+id); err == nil {
				p.TaskToken = string(tok)
			}
		}
	}
	return p, nil
}

// owner rebuilds the plan owner's caller so the task is issued under their current rights, not the approver's.
func (s *Service) owner(ctx context.Context, ownerType, ownerID, project string) (*authn.Caller, error) {
	if ownerType == string(principal.ServiceAccount) {
		g, err := authz.LoadServiceAccount(ctx, s.Pool, ownerID)
		if err != nil {
			return nil, err
		}
		return &authn.Caller{P: &principal.Principal{Type: principal.ServiceAccount, ID: ownerID, ProjectID: project}, Grants: g,
			Priority: levels.Standard, MaxPriority: levels.Standard}, nil
	}
	u, err := directory.GetUser(ctx, s.Pool, ownerID)
	if err != nil || u.Status != directory.Active {
		return nil, httpx.Errorf(http.StatusConflict, "owner_inactive", "the plan's owner is no longer active")
	}
	groups, err := directory.GroupIDs(ctx, s.Pool, u.ID)
	if err != nil {
		return nil, err
	}
	g, err := authz.LoadUser(ctx, s.Pool, u.ID, groups)
	if err != nil {
		return nil, err
	}
	if !g.Can(authz.Tasks, project) {
		return nil, httpx.Errorf(http.StatusConflict, "owner_inactive", "the plan's owner can no longer run tasks in this project")
	}
	return &authn.Caller{P: &principal.Principal{Type: principal.User, ID: u.ID, UserID: u.ID, Email: u.Email, ProjectID: project, Groups: groups},
		Grants: g, Priority: levels.Standard, MaxPriority: levels.Interactive}, nil
}
