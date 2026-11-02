package plans

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/tasks"
)

// Register mounts the plan API.
func (s *Service) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/starlight/plans", s.create)
	mux.HandleFunc("GET /v1/starlight/plans/{id}", s.get)
	mux.HandleFunc("POST /v1/starlight/plans/{id}/approve", s.decide("approved"))
	mux.HandleFunc("POST /v1/starlight/plans/{id}/reject", s.decide("rejected"))
}

func (s *Service) caller(r *http.Request) (*authn.Caller, error) {
	c, err := s.Auth.Authenticate(r.Context(), httpx.BearerToken(r), r.Header.Get("X-Starlight-Project"), s.Proxies.ClientIP(r))
	if err != nil {
		return nil, httpx.ErrUnauthenticated
	}
	return c, nil
}

func (s *Service) create(w http.ResponseWriter, r *http.Request) {
	c, err := s.caller(r)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	if c.Task != nil || !c.P.HasScope("inference") || !c.Grants.Can(authz.Plans, c.P.ProjectID) {
		httpx.Write(w, httpx.OpenAI, httpx.ErrForbidden)
		return
	}
	var d Draft
	dec := json.NewDecoder(io.LimitReader(r.Body, 8<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		httpx.Write(w, httpx.OpenAI, bad("body must be a plan draft"))
		return
	}
	if err := d.validate(); err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	if err := s.localExecutor(r.Context(), c, d.Executor); err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	body, messages, err := d.egress()
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	id := ids.New()
	conc := max(d.MaxConcurrency, 1)
	ttl := d.TTLSeconds
	if ttl <= 0 {
		ttl = 86400
	}
	if _, err := s.Pool.Exec(r.Context(), `INSERT INTO plans (id, project_id, owner_type, owner_id, title, planner_model, executor_model, egress_sha256,
		egress_bytes, status, budget_tokens, max_concurrency, max_depth, ttl_seconds, api_key_id, refresh_family)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'drafting',$10,$11,$12,$13,$14,$15)`,
		id, c.P.ProjectID, string(c.P.Type), c.P.ID, d.Title, d.Planner, d.Executor, digest(messages), len(messages), d.BudgetTokens, conc, d.MaxDepth,
		ttl, nullable(c.P.KeyID), nullable(c.Family)); err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	s.record(r, c, "plan.egress", id, map[string]any{"planner": d.Planner, "messages_sha256": hexOf(digest(messages)), "bytes": len(messages),
		"prompt": PromptVersion})
	text, err := s.complete(r, body)
	if err != nil {
		_, _ = s.Pool.Exec(context.WithoutCancel(r.Context()), "UPDATE plans SET status = 'failed', error = $2, updated_at = now() WHERE id = $1", id, err.Error())
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	sealed, err := s.Keys.Seal([]byte(text), "plans.plan:"+id)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	if _, err := s.Pool.Exec(r.Context(), "UPDATE plans SET plan_sealed = $2, plan_sha256 = $3, status = 'pending_approval', updated_at = now() WHERE id = $1",
		id, sealed, digest([]byte(text))); err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	var mode string
	_ = s.Pool.QueryRow(r.Context(), "SELECT plan_approval FROM projects WHERE id = $1", c.P.ProjectID).Scan(&mode)
	if mode == "auto" {
		if err := s.approve(r, c, id, hexOf(digest([]byte(text))), true); err != nil {
			httpx.Write(w, httpx.OpenAI, err)
			return
		}
	}
	p, err := s.view(r.Context(), c, id)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	httpx.JSON(w, http.StatusCreated, p)
}

func (s *Service) decide(status string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c, err := s.caller(r)
		if err != nil {
			httpx.Write(w, httpx.OpenAI, err)
			return
		}
		var in struct {
			PlanSHA256 string `json:"plan_sha256"`
		}
		_ = json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&in)
		id := r.PathValue("id")
		if status == "approved" {
			err = s.approve(r, c, id, in.PlanSHA256, false)
		} else {
			err = s.reject(r, c, id)
		}
		if err != nil {
			httpx.Write(w, httpx.OpenAI, err)
			return
		}
		p, err := s.view(r.Context(), c, id)
		if err != nil {
			httpx.Write(w, httpx.OpenAI, err)
			return
		}
		httpx.JSON(w, http.StatusOK, p)
	}
}

// approve checks the approver saw exactly this plan, then issues the owner a pinned, local-only task.
func (s *Service) approve(r *http.Request, c *authn.Caller, id, sum string, auto bool) error {
	var project, ownerType, ownerID, executor, status string
	var planSum []byte
	var budget *int64
	var conc, depth, ttl int
	var keyID, family *string
	err := s.Pool.QueryRow(r.Context(), `SELECT project_id, owner_type, owner_id, executor_model, status, plan_sha256, budget_tokens, max_concurrency,
		max_depth, ttl_seconds, api_key_id, refresh_family FROM plans WHERE id = $1`, id).
		Scan(&project, &ownerType, &ownerID, &executor, &status, &planSum, &budget, &conc, &depth, &ttl, &keyID, &family)
	if err != nil || !validID(id) || !auto && !canApprove(c, project) {
		return httpx.ErrNotFound
	}
	want, _ := hex.DecodeString(sum)
	if status != "pending_approval" || len(want) != 32 || !equal(want, planSum) {
		return httpx.Errorf(http.StatusConflict, "plan_changed", "approve with the plan_sha256 of the plan you reviewed, while it is pending")
	}
	owner, err := s.owner(r.Context(), ownerType, ownerID, project)
	if err != nil {
		return err
	}
	if keyID != nil {
		owner.P.KeyID = *keyID
	}
	if family != nil {
		owner.Family = *family
	}
	local := true
	task, err := s.Tasks.Create(r.Context(), owner, tasks.Spec{Name: "plan " + id[:8], Models: []string{executor}, LocalOnly: &local,
		MaxConcurrency: conc, MaxDepth: depth, BudgetTokens: budget, TTLSeconds: ttl, PinRevisions: true, PlanID: id})
	if err != nil {
		return err
	}
	sealed, err := s.Keys.Seal([]byte(task.Token), "plans.task_token:"+id)
	if err != nil {
		return err
	}
	return db.Tx(r.Context(), s.Pool, func(tx pgx.Tx) error {
		tag, err := tx.Exec(r.Context(), `UPDATE plans SET status = 'approved', approved_by = $2, approved_at = now(), task_id = $3, task_token_sealed = $4,
			updated_at = now() WHERE id = $1 AND status = 'pending_approval'`, id, approver(c, auto), task.ID, sealed)
		if err != nil || tag.RowsAffected() != 1 {
			return httpx.Errorf(http.StatusConflict, "plan_changed", "the plan was decided by someone else")
		}
		return audit.Append(r.Context(), tx, s.event(r, c, "plan.approved", id, map[string]any{"plan_sha256": sum, "task_id": task.ID, "auto": auto}))
	})
}

// Approve is the console's way to approve a plan, with the same checks as the API.
func (s *Service) Approve(r *http.Request, c *authn.Caller, id, sum string) error {
	return s.approve(r, c, id, sum, false)
}

// Reject is the console's way to reject a plan, with the same checks as the API.
func (s *Service) Reject(r *http.Request, c *authn.Caller, id string) error {
	return s.reject(r, c, id)
}

func nullable(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func approver(c *authn.Caller, auto bool) *string {
	if auto || c.P.UserID == "" {
		return nil
	}
	return &c.P.UserID
}

func equal(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	var v byte
	for i := range a {
		v |= a[i] ^ b[i]
	}
	return v == 0
}

func (s *Service) reject(r *http.Request, c *authn.Caller, id string) error {
	var project string
	if err := s.Pool.QueryRow(r.Context(), "SELECT project_id FROM plans WHERE id = $1", id).Scan(&project); err != nil || !canApprove(c, project) {
		return httpx.ErrNotFound
	}
	return db.Tx(r.Context(), s.Pool, func(tx pgx.Tx) error {
		tag, err := tx.Exec(r.Context(), "UPDATE plans SET status = 'rejected', updated_at = now() WHERE id = $1 AND status = 'pending_approval'", id)
		if err != nil || tag.RowsAffected() != 1 {
			return err
		}
		return audit.Append(r.Context(), tx, s.event(r, c, "plan.rejected", id, nil))
	})
}
