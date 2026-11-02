package gateway

import (
	"context"
	"net/http"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/guard"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

// Project is the policy a project imposes on its requests.
type Project struct {
	ID           string
	Slug         string
	Class        levels.Class
	MaxPriority  levels.Priority
	SecretsGuard string
	Isolated     bool
}

// Policies caches project policy and installation settings for a few seconds.
type Policies struct {
	Pool     *pgxpool.Pool
	TTL      time.Duration
	mu       sync.Mutex
	projects map[string]cachedProject
	offline  bool
	offAt    time.Time
}

type cachedProject struct {
	p       Project
	expires time.Time
}

func (p *Policies) ttl() time.Duration {
	if p.TTL <= 0 {
		return 5 * time.Second
	}
	return p.TTL
}

// Project loads a project's policy.
func (p *Policies) Project(ctx context.Context, id string) (Project, error) {
	p.mu.Lock()
	if c, ok := p.projects[id]; ok && time.Now().Before(c.expires) {
		p.mu.Unlock()
		return c.p, nil
	}
	p.mu.Unlock()
	var pr Project
	var class, prio string
	err := p.Pool.QueryRow(ctx, `SELECT id, slug, classification, max_priority, secrets_guard, isolate_prompt_cache FROM projects
		WHERE id = $1 AND archived_at IS NULL`, id).Scan(&pr.ID, &pr.Slug, &class, &prio, &pr.SecretsGuard, &pr.Isolated)
	if err != nil {
		return pr, err
	}
	pr.Class, _ = levels.ParseClass(class)
	pr.MaxPriority, _ = levels.ParsePriority(prio)
	p.mu.Lock()
	if p.projects == nil {
		p.projects = map[string]cachedProject{}
	}
	p.projects[id] = cachedProject{pr, time.Now().Add(p.ttl())}
	p.mu.Unlock()
	return pr, nil
}

// Offline reports whether the installation forbids cloud egress; on a database error it fails closed.
func (p *Policies) Offline(ctx context.Context) bool {
	p.mu.Lock()
	if time.Now().Before(p.offAt) {
		v := p.offline
		p.mu.Unlock()
		return v
	}
	p.mu.Unlock()
	off := true
	if err := p.Pool.QueryRow(ctx, "SELECT offline FROM settings WHERE id").Scan(&off); err != nil {
		off = true
	}
	p.mu.Lock()
	p.offline, p.offAt = off, time.Now().Add(p.ttl())
	p.mu.Unlock()
	return off
}

// Flush forgets cached policy.
func (p *Policies) Flush() {
	p.mu.Lock()
	p.projects, p.offAt = nil, time.Time{}
	p.mu.Unlock()
}

// classify returns the strictest of the project default, the credential floor and a client request to raise it.
func classify(r *http.Request, pr Project, c *authn.Caller) levels.Class {
	class := levels.Max(pr.Class, c.ClassFloor)
	if h := r.Header.Get("X-Starlight-Classification"); h != "" {
		if asked, err := levels.ParseClass(h); err == nil {
			class = levels.Max(class, asked)
		}
	}
	return class
}

// prioritise lets a client lower its priority freely and raise it only to what its credential and project allow.
func prioritise(r *http.Request, in Info, pr Project, c *authn.Caller) levels.Priority {
	p := c.Priority
	if h := r.Header.Get("X-Starlight-Priority"); h != "" {
		if asked, err := levels.ParsePriority(h); err == nil {
			p = asked
		}
	}
	if in.BackgroundIn {
		p = levels.Batch
	}
	return p.Clamp(c.MaxPriority).Clamp(pr.MaxPriority)
}

// plan decides where a request may run: route policy, offline mode, task scope, classification and protocol.
type plan struct {
	local      bool
	cloud      []catalog.ProviderRoute
	cloudFirst bool
	class      levels.Class
	findings   []guard.Finding
}

func decide(m catalog.Model, routes []catalog.ProviderRoute, protocol string, class levels.Class, offline bool, c *authn.Caller) plan {
	pl := plan{class: class}
	pl.local = m.RoutePolicy != "cloud_only"
	pl.cloudFirst = m.RoutePolicy == "cloud_first" || m.RoutePolicy == "cloud_only"
	if m.RoutePolicy == "local_only" || offline || c.Task != nil && c.Task.LocalOnly {
		return pl
	}
	for _, r := range routes {
		max, err := levels.ParseClass(r.MaxClass)
		if err == nil && class <= max && r.Speaks(protocol) {
			pl.cloud = append(pl.cloud, r)
		}
	}
	return pl
}

// screen applies the project's credential rule: "local" keeps such requests on local routes, "block" refuses them.
func screen(pl plan, pr Project, raw []byte) (plan, error) {
	if pr.SecretsGuard == "off" || pr.SecretsGuard == "local" && len(pl.cloud) == 0 {
		return pl, nil
	}
	pl.findings = guard.ScanJSON(raw)
	if len(pl.findings) == 0 {
		return pl, nil
	}
	if pr.SecretsGuard == "block" {
		return pl, httpx.Errorf(http.StatusBadRequest, "credentials_detected", "the request contains what look like credentials and this project refuses such requests")
	}
	pl.class = levels.Restricted
	kept := pl.cloud[:0]
	for _, r := range pl.cloud {
		if max, err := levels.ParseClass(r.MaxClass); err == nil && max >= levels.Restricted {
			kept = append(kept, r)
		}
	}
	pl.cloud = kept
	if !pl.local && len(pl.cloud) == 0 {
		return pl, httpx.Errorf(http.StatusBadRequest, "credentials_detected", "the request contains what look like credentials and this model has no local route")
	}
	return pl, nil
}
