// Package plans runs frontier-planned, locally executed work: a frontier drafts, an approver signs, local models build.
package plans

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/tasks"
)

// PlannerPrompt is the fixed system prompt; its version is recorded with every plan.
const PlannerPrompt = "You are the planning stage of a two-stage engineering workflow. A model inside the organisation's network will " +
	"execute your plan without access to you. Write a complete, self-contained plan in Markdown with these sections: Goal, " +
	"Assumptions, Steps (numbered; each names the files to change and how to check it), Tests, Risks. State assumptions instead " +
	"of asking questions. Never include credentials."

// PromptVersion identifies PlannerPrompt in audit records.
const PromptVersion = "planner-v1"

// Attachment is context the requester explicitly chose to send to the planner.
type Attachment struct {
	Name    string `json:"name"`
	Content string `json:"content"`
}

// Draft asks for a plan.
type Draft struct {
	Title          string       `json:"title"`
	Planner        string       `json:"planner_model"`
	Executor       string       `json:"executor_model"`
	Brief          string       `json:"brief"`
	Context        []Attachment `json:"context"`
	MaxTokens      int          `json:"max_tokens"`
	BudgetTokens   *int64       `json:"budget_tokens"`
	MaxConcurrency int          `json:"max_concurrency"`
	MaxDepth       int          `json:"max_depth"`
	TTLSeconds     int          `json:"ttl_seconds"`
}

// Plan is the API view of a plan.
type Plan struct {
	ID           string  `json:"id"`
	ProjectID    string  `json:"project_id"`
	Title        string  `json:"title"`
	Planner      string  `json:"planner_model"`
	Executor     string  `json:"executor_model"`
	Status       string  `json:"status"`
	Text         string  `json:"plan,omitempty"`
	PlanSHA256   string  `json:"plan_sha256,omitempty"`
	EgressSHA256 string  `json:"egress_sha256"`
	EgressBytes  int     `json:"egress_bytes"`
	TaskID       *string `json:"task_id,omitempty"`
	TaskToken    string  `json:"task_token,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// Service drafts, approves and serves plans.
type Service struct {
	Pool    *pgxpool.Pool
	Keys    *keyring.Keyring
	Auth    *authn.Authenticator
	Catalog *catalog.Catalog
	Tasks   *tasks.Service
	Proxies *httpx.Proxies
	Chat    http.Handler
}

func bad(msg string) error { return httpx.Errorf(http.StatusBadRequest, "invalid_request_error", msg) }

func (d Draft) validate() error {
	switch {
	case strings.TrimSpace(d.Title) == "" || len(d.Title) > 200:
		return bad("title is required and at most 200 characters")
	case d.Planner == "" || d.Executor == "":
		return bad("planner_model and executor_model are required")
	case strings.TrimSpace(d.Brief) == "":
		return bad("brief is required")
	case len(d.Context) > 32:
		return bad("at most 32 context attachments")
	case d.MaxDepth < 0 || d.MaxDepth > tasks.MaxDepth:
		return bad("max_depth must be between 0 and 5")
	}
	return nil
}

// egress builds the planner request; messages are its exact content bytes, which routing never rewrites.
func (d Draft) egress() (body, messages []byte, err error) {
	var user strings.Builder
	user.WriteString("# Brief\n\n" + d.Brief + "\n")
	for _, a := range d.Context {
		user.WriteString("\n# Attachment: " + a.Name + "\n\n" + a.Content + "\n")
	}
	maxTokens := d.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 8192
	}
	messages, err = json.Marshal([]map[string]string{{"role": "system", "content": PlannerPrompt}, {"role": "user", "content": user.String()}})
	if err != nil {
		return nil, nil, err
	}
	body, err = json.Marshal(map[string]any{"model": d.Planner, "max_tokens": maxTokens, "messages": json.RawMessage(messages)})
	return body, messages, err
}

// localExecutor requires the executor to resolve for the project and have a local deployment.
func (s *Service) localExecutor(ctx context.Context, c *authn.Caller, name string) error {
	m, _, err := s.Catalog.Resolve(ctx, c.P.ProjectID, name)
	if err != nil || !c.AllowsModel(m.Name) && !c.AllowsModel(name) {
		return bad("executor_model is not available to this credential")
	}
	var ok bool
	if err := s.Pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM deployments WHERE model_id = $1 AND status = 'active')", m.ID).Scan(&ok); err != nil {
		return err
	}
	if !ok || m.RoutePolicy == "cloud_only" {
		return bad("executor_model must have a local deployment")
	}
	return nil
}

// complete sends the planner request through the gateway with the requester's own credential and policy.
func (s *Service) complete(r *http.Request, body []byte) (string, error) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	for _, h := range []string{"Authorization", "X-Api-Key", "X-Starlight-Project", "X-Starlight-Classification", "X-Forwarded-For"} {
		if v := r.Header.Get(h); v != "" {
			req.Header.Set(h, v)
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Starlight-Client", "starlight/plans")
	req.RemoteAddr = r.RemoteAddr
	rec := httptest.NewRecorder()
	s.Chat.ServeHTTP(rec, req)
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		return "", errors.New("the planner returned an unreadable response")
	}
	if rec.Code != http.StatusOK || out.Error != nil || len(out.Choices) == 0 {
		msg := "the planner request failed"
		if out.Error != nil {
			msg = out.Error.Message
		}
		return "", httpx.Errorf(rec.Code, "planner_failed", msg)
	}
	return out.Choices[0].Message.Content, nil
}

func digest(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}

func (s *Service) record(r *http.Request, c *authn.Caller, action, planID string, details map[string]any) {
	_ = audit.Log(context.WithoutCancel(r.Context()), s.Pool, s.event(r, c, action, planID, details))
}

func (s *Service) event(r *http.Request, c *authn.Caller, action, planID string, details map[string]any) audit.Event {
	e := audit.FromRequest(r, s.Proxies, c.P)
	e.Action, e.TargetType, e.TargetID, e.Details = action, "plan", planID, details
	return e
}

func canApprove(c *authn.Caller, projectID string) bool {
	return c.Task == nil && c.Grants.Can(authz.ProjectApprove, projectID)
}

func hexOf(b []byte) string { return hex.EncodeToString(b) }

func validID(id string) bool { return ids.Valid(id) }
