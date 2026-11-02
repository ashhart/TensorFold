package authn

import (
	"context"
	"encoding/json"
	"errors"
	"net/netip"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/principal"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

func (a *Authenticator) apiKey(ctx context.Context, token string, ip netip.Addr) (*Caller, error) {
	var keyID, ownerType, ownerID, projectID, priority string
	var models []string
	var class *string
	var cidrs []netip.Prefix
	var expires, lastUsed *time.Time
	err := a.Pool.QueryRow(ctx, `SELECT k.id, k.owner_type, k.owner_id, k.project_id, k.models, k.priority, k.classification, k.allowed_cidrs,
		k.expires_at, k.last_used_at FROM api_keys k JOIN projects p ON p.id = k.project_id
		WHERE k.secret_hash = $1 AND p.archived_at IS NULL AND (k.expires_at IS NULL OR k.expires_at > now())
		AND (k.revoked_at IS NULL) AND (k.grace_until IS NULL OR k.grace_until > now())`, secret.Hash(token)).
		Scan(&keyID, &ownerType, &ownerID, &projectID, &models, &priority, &class, &cidrs, &expires, &lastUsed)
	if db.NotFound(err) {
		return nil, ErrInvalid
	}
	if err != nil {
		return nil, err
	}
	if len(cidrs) > 0 && !inAny(ip, cidrs) {
		return nil, ErrInvalid
	}
	var c *Caller
	switch ownerType {
	case "user":
		c, err = a.user(ctx, ownerID, projectID, principal.ViaAPIKey)
	default:
		c, err = a.serviceAccount(ctx, ownerID, projectID)
	}
	if err != nil {
		return nil, err
	}
	c.P.KeyID = keyID
	c.P.Scope = []string{"inference"}
	c.Models = models
	if p, err := levels.ParsePriority(priority); err == nil {
		c.Priority, c.MaxPriority = p, p
	}
	if class != nil {
		c.ClassFloor, _ = levels.ParseClass(*class)
	}
	if expires != nil {
		c.expires, c.KeyExpires = *expires, expires
	}
	if lastUsed == nil || time.Since(*lastUsed) > time.Minute {
		uctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), a.timeout())
		_, _ = a.Pool.Exec(uctx, "UPDATE api_keys SET last_used_at = now() WHERE id = $1", keyID)
		cancel()
	}
	return c, nil
}

func inAny(ip netip.Addr, cidrs []netip.Prefix) bool {
	for _, p := range cidrs {
		if p.Contains(ip) {
			return true
		}
	}
	return false
}

func (a *Authenticator) access(ctx context.Context, token, hint string) (*Caller, error) {
	var userID, scope string
	var family *string
	var expires time.Time
	err := a.Pool.QueryRow(ctx, `SELECT user_id, scope, family_id, expires_at FROM access_tokens
		WHERE token_hash = $1 AND revoked_at IS NULL AND expires_at > now()`, secret.Hash(token)).Scan(&userID, &scope, &family, &expires)
	if db.NotFound(err) {
		return nil, ErrInvalid
	}
	if err != nil {
		return nil, err
	}
	c, err := a.user(ctx, userID, hint, principal.ViaAccess)
	if err != nil {
		return nil, err
	}
	c.P.Scope = splitScope(scope)
	c.Priority, c.MaxPriority = levels.Interactive, levels.Interactive
	c.expires = expires
	if family != nil {
		c.Family = *family
	}
	return c, nil
}

func splitScope(s string) []string {
	var out []string
	start := -1
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ' ' {
			if start >= 0 {
				out = append(out, s[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	return out
}

// user builds a caller for an active user acting in a project named by id or slug, or their only project.
func (a *Authenticator) user(ctx context.Context, userID, hint string, via principal.Credential) (*Caller, error) {
	u, err := directory.GetUser(ctx, a.Pool, userID)
	if err != nil && !errors.Is(err, directory.ErrNotFound) {
		return nil, err
	}
	if err != nil || u.Status != directory.Active {
		return nil, ErrInvalid
	}
	groups, err := directory.GroupIDs(ctx, a.Pool, u.ID)
	if err != nil {
		return nil, err
	}
	g, err := authz.LoadUser(ctx, a.Pool, u.ID, groups)
	if err != nil {
		return nil, err
	}
	project, err := a.pickProject(ctx, g, hint)
	if err != nil {
		return nil, err
	}
	p := &principal.Principal{Type: principal.User, ID: u.ID, UserID: u.ID, Email: u.Email, DisplayName: u.DisplayName, ProjectID: project,
		Groups: groups, Via: via}
	return &Caller{P: p, Grants: g, Priority: levels.Standard, MaxPriority: levels.Interactive}, nil
}

func (a *Authenticator) pickProject(ctx context.Context, g authz.Grants, hint string) (string, error) {
	usable := g.ProjectIDs(authz.Inference)
	if hint != "" {
		id := hint
		if !ids.Valid(hint) {
			err := a.Pool.QueryRow(ctx, "SELECT id FROM projects WHERE slug = $1 AND archived_at IS NULL", hint).Scan(&id)
			if db.NotFound(err) {
				return "", ErrProject
			}
			if err != nil {
				return "", err
			}
		}
		if g.Can(authz.Inference, id) {
			return id, nil
		}
		return "", ErrProject
	}
	if len(usable) == 1 {
		return usable[0], nil
	}
	var everyone string
	err := a.Pool.QueryRow(ctx, "SELECT id FROM projects WHERE everyone AND archived_at IS NULL ORDER BY created_at LIMIT 1").Scan(&everyone)
	if err != nil && !db.NotFound(err) {
		return "", err
	}
	if err == nil && g.Can(authz.Inference, everyone) {
		return everyone, nil
	}
	return "", ErrProject
}

func (a *Authenticator) serviceAccount(ctx context.Context, id, project string) (*Caller, error) {
	var status, name, saProject string
	err := a.Pool.QueryRow(ctx, "SELECT status, name, project_id FROM service_accounts WHERE id = $1", id).Scan(&status, &name, &saProject)
	if err != nil && !db.NotFound(err) {
		return nil, err
	}
	if err != nil || status != "active" || saProject != project {
		return nil, ErrInvalid
	}
	g, err := authz.LoadServiceAccount(ctx, a.Pool, id)
	if err != nil {
		return nil, err
	}
	p := &principal.Principal{Type: principal.ServiceAccount, ID: id, DisplayName: name, ProjectID: project, Via: principal.ViaAPIKey}
	return &Caller{P: p, Grants: g, Priority: levels.Standard, MaxPriority: levels.Standard}, nil
}

// task resolves a task token; the root's minting key or Connect sign-in keeps constraining every task in the tree.
func (a *Authenticator) task(ctx context.Context, token string, ip netip.Addr) (*Caller, error) {
	var t Task
	var ownerType, ownerID, project string
	var models []string
	var pins []byte
	var keyID, family *string
	err := a.Pool.QueryRow(ctx, `SELECT t.id, t.root_id, t.depth, t.max_depth, t.max_concurrency, r.max_concurrency, t.local_only, t.pins, r.budget_tokens,
		t.expires_at, t.owner_type, t.owner_id, t.project_id, t.models, r.api_key_id, r.refresh_family
		FROM tasks t JOIN tasks r ON r.id = t.root_id
		WHERE t.token_hash = $1 AND t.revoked_at IS NULL AND t.expires_at > now()
		AND NOT EXISTS (SELECT 1 FROM tasks a WHERE a.id = ANY(t.ancestors) AND (a.revoked_at IS NOT NULL OR a.expires_at <= now()))`,
		secret.Hash(token)).Scan(&t.ID, &t.RootID, &t.Depth, &t.MaxDepth, &t.MaxConcurrency, &t.TreeConcurrency, &t.LocalOnly, &pins, &t.BudgetTokens,
		&t.ExpiresAt, &ownerType, &ownerID, &project, &models, &keyID, &family)
	if db.NotFound(err) {
		return nil, ErrInvalid
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(pins, &t.Pins); err != nil {
		return nil, err
	}
	var c *Caller
	if ownerType == "user" {
		c, err = a.user(ctx, ownerID, project, principal.ViaTask)
	} else {
		c, err = a.serviceAccount(ctx, ownerID, project)
	}
	if err != nil {
		return nil, err
	}
	c.Priority, c.MaxPriority = levels.Standard, levels.Standard
	if keyID != nil {
		if err := a.mintedByKey(ctx, c, *keyID, ip); err != nil {
			return nil, err
		}
	}
	if family != nil {
		var revoked bool
		if err := a.Pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM refresh_tokens WHERE family_id = $1 AND revoked_at IS NOT NULL)", *family).
			Scan(&revoked); err != nil {
			return nil, err
		}
		if revoked {
			return nil, ErrInvalid
		}
	}
	c.P.Via, c.P.TaskID, c.P.TaskRootID, c.P.Scope = principal.ViaTask, t.ID, t.RootID, []string{"inference"}
	c.Models = intersect(c.Models, models)
	c.Task, c.expires = &t, t.ExpiresAt
	return c, nil
}

// mintedByKey applies the API key that created the task tree: it must still be live, and its limits carry over.
func (a *Authenticator) mintedByKey(ctx context.Context, c *Caller, keyID string, ip netip.Addr) error {
	var priority string
	var models []string
	var class *string
	var cidrs []netip.Prefix
	err := a.Pool.QueryRow(ctx, `SELECT priority, models, classification, allowed_cidrs FROM api_keys WHERE id = $1 AND revoked_at IS NULL
		AND (expires_at IS NULL OR expires_at > now()) AND (grace_until IS NULL OR grace_until > now())`, keyID).Scan(&priority, &models, &class, &cidrs)
	if db.NotFound(err) {
		return ErrInvalid
	}
	if err != nil {
		return err
	}
	if len(cidrs) > 0 && !inAny(ip, cidrs) {
		return ErrInvalid
	}
	c.P.KeyID, c.Models = keyID, models
	if p, err := levels.ParsePriority(priority); err == nil {
		c.Priority, c.MaxPriority = min(c.Priority, p), min(c.MaxPriority, p)
	}
	if class != nil {
		c.ClassFloor, _ = levels.ParseClass(*class)
	}
	return nil
}

// intersect narrows a credential's model list by a task's; nil means unrestricted.
func intersect(credential, task []string) []string {
	if credential == nil {
		return task
	}
	var out []string
	for _, m := range task {
		for _, k := range credential {
			if m == k {
				out = append(out, m)
				break
			}
		}
	}
	if out == nil {
		out = []string{}
	}
	return out
}
