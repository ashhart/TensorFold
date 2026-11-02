package tasks

import (
	"context"
	"encoding/json"
	"io"
	"net/http"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Register mounts the task API.
func (s *Service) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/starlight/tasks", s.create)
	mux.HandleFunc("GET /v1/starlight/tasks/{id}", s.get)
	mux.HandleFunc("DELETE /v1/starlight/tasks/{id}", s.revoke)
}

func (s *Service) caller(r *http.Request) (*authn.Caller, error) {
	return s.Auth.Authenticate(r.Context(), httpx.BearerToken(r), r.Header.Get("X-Starlight-Project"), s.Proxies.ClientIP(r))
}

func (s *Service) create(w http.ResponseWriter, r *http.Request) {
	c, err := s.caller(r)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, httpx.ErrUnauthenticated)
		return
	}
	var spec Spec
	dec := json.NewDecoder(io.LimitReader(r.Body, 64<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&spec); err != nil {
		httpx.Write(w, httpx.OpenAI, bad("body must be a task specification"))
		return
	}
	t, err := s.Create(r.Context(), c, spec)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	httpx.JSON(w, http.StatusCreated, t)
}

// visible reports whether the caller owns the task or holds a task token at or above it.
func (s *Service) visible(ctx context.Context, c *authn.Caller, id string) (bool, error) {
	var ownerType, ownerID, root string
	var ancestors []string
	err := s.Pool.QueryRow(ctx, "SELECT owner_type, owner_id, root_id, ancestors FROM tasks WHERE id = $1", id).Scan(&ownerType, &ownerID, &root, &ancestors)
	if err != nil {
		return false, nil
	}
	if c.Task != nil {
		if c.Task.ID == id {
			return true, nil
		}
		for _, a := range ancestors {
			if a == c.Task.ID {
				return true, nil
			}
		}
		return false, nil
	}
	return ownerType == string(c.P.Type) && ownerID == c.P.ID, nil
}

func (s *Service) get(w http.ResponseWriter, r *http.Request) {
	c, err := s.caller(r)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, httpx.ErrUnauthenticated)
		return
	}
	id := r.PathValue("id")
	if ok, _ := s.visible(r.Context(), c, id); !ids.Valid(id) || !ok {
		httpx.Write(w, httpx.OpenAI, httpx.ErrNotFound)
		return
	}
	t, err := s.load(r.Context(), id)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	httpx.JSON(w, http.StatusOK, t)
}

func (s *Service) load(ctx context.Context, id string) (Task, error) {
	var t Task
	var pins []byte
	var revoked *string
	err := s.Pool.QueryRow(ctx, `SELECT t.id, t.root_id, t.parent_id, t.depth, t.name, t.models, t.local_only, t.max_concurrency, t.max_depth,
		r.budget_tokens, t.pins, t.expires_at, t.revoked_at::text,
		coalesce((SELECT sum(u.prompt_tokens + u.completion_tokens) FROM usage_events u WHERE u.task_id IN
			(SELECT x.id FROM tasks x WHERE x.id = t.id OR t.id = ANY(x.ancestors))), 0)
		FROM tasks t JOIN tasks r ON r.id = t.root_id WHERE t.id = $1`, id).
		Scan(&t.ID, &t.RootID, &t.ParentID, &t.Depth, &t.Name, &t.Models, &t.LocalOnly, &t.MaxConcurrency, &t.MaxDepth, &t.BudgetTokens,
			&pins, &t.ExpiresAt, &revoked, &t.UsedTokens)
	if err != nil {
		return t, err
	}
	_ = json.Unmarshal(pins, &t.Pins)
	t.Revoked = revoked != nil
	return t, nil
}

func (s *Service) revoke(w http.ResponseWriter, r *http.Request) {
	c, err := s.caller(r)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, httpx.ErrUnauthenticated)
		return
	}
	id := r.PathValue("id")
	if ok, _ := s.visible(r.Context(), c, id); !ids.Valid(id) || !ok {
		httpx.Write(w, httpx.OpenAI, httpx.ErrNotFound)
		return
	}
	n, err := Revoke(r.Context(), s, id)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	e := audit.FromRequest(r, s.Proxies, c.P)
	e.Action, e.TargetType, e.TargetID, e.Details = "task.revoked", "task", id, map[string]any{"revoked": n}
	_ = audit.Log(context.WithoutCancel(r.Context()), s.Pool, e)
	s.Auth.Forget()
	httpx.JSON(w, http.StatusOK, map[string]any{"id": id, "revoked": n})
}

// Revoke ends a task and every task below it.
func Revoke(ctx context.Context, s *Service, id string) (int64, error) {
	tag, err := s.Pool.Exec(ctx, "UPDATE tasks SET revoked_at = now() WHERE (id = $1 OR $1 = ANY(ancestors)) AND revoked_at IS NULL", id)
	return tag.RowsAffected(), err
}
