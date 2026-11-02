package gateway

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/netip"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/guard"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
	"github.com/ashhart/TensorFold/starlight/internal/metrics"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

// Gateway serves /v1 inference under authentication, policy, admission and accounting.
type Gateway struct {
	Pool      *pgxpool.Pool
	Auth      *authn.Authenticator
	Catalog   *catalog.Catalog
	Admitter  *admission.Admitter
	Policies  *Policies
	Nodes     *Nodes
	Providers *providers.Client
	Metrics   *metrics.Set
	Proxies   *httpx.Proxies
	Config    config.Gateway
	Log       *slog.Logger
	limitsMu  sync.Mutex
	limits    map[string]cachedLimits
}

// bodyDeadline bounds how long a client may take to send its request body.
const bodyDeadline = 2 * time.Minute

type cachedLimits struct {
	l       []admission.Limit
	expires time.Time
}

// Register mounts the inference API.
func (g *Gateway) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/models", g.models)
	mux.HandleFunc("POST /v1/chat/completions", g.infer(providers.ChatCompletions, "/v1/chat/completions"))
	mux.HandleFunc("POST /v1/completions", g.infer(providers.Completions, "/v1/completions"))
	mux.HandleFunc("POST /v1/embeddings", g.infer(providers.Embeddings, "/v1/embeddings"))
	mux.HandleFunc("POST /v1/responses", g.infer(providers.Responses, "/v1/responses"))
	mux.HandleFunc("GET /v1/responses/{id}", g.storedResponse)
	mux.HandleFunc("DELETE /v1/responses/{id}", g.storedResponse)
	mux.HandleFunc("POST /v1/messages", g.infer(providers.Messages, "/v1/messages"))
	mux.HandleFunc("POST /v1/messages/count_tokens", g.infer(providers.CountTokens, "/v1/messages/count_tokens"))
}

func dialectFor(protocol string) httpx.Dialect {
	if protocol == providers.Messages || protocol == providers.CountTokens {
		return httpx.Anthropic
	}
	return httpx.OpenAI
}

// call is one inference request after every check that precedes admission.
type call struct {
	protocol   string
	enginePath string
	dialect    httpx.Dialect
	caller     *authn.Caller
	token      string
	hint       string
	ip         netip.Addr
	project    Project
	model      catalog.Model
	info       Info
	body       *Body
	plan       plan
	req        admission.Request
	client     string
	requestID  string
	headers    http.Header
}

func (g *Gateway) infer(protocol, enginePath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rc := http.NewResponseController(w)
		_ = rc.SetReadDeadline(start.Add(bodyDeadline))
		c, err := g.prepare(r, protocol, enginePath)
		_ = rc.SetReadDeadline(time.Time{})
		if err != nil {
			g.refuse(w, dialectFor(protocol), err)
			return
		}
		g.serve(w, r, c, start)
	}
}

func (g *Gateway) authenticate(r *http.Request) (*authn.Caller, string, string, netip.Addr, error) {
	ip := g.Proxies.ClientIP(r)
	token, hint := httpx.BearerToken(r), strings.TrimSpace(r.Header.Get("X-Starlight-Project"))
	c, err := g.Auth.Authenticate(r.Context(), token, hint, ip)
	if err != nil {
		return nil, "", "", ip, err
	}
	if !mayInfer(c, c.P.ProjectID) {
		return nil, "", "", ip, httpx.Errorf(http.StatusForbidden, "permission_denied", "this credential may not run inference in the project")
	}
	return c, token, hint, ip, nil
}

// mayInfer reports whether a caller's credential and grants let it run inference in project.
func mayInfer(c *authn.Caller, project string) bool {
	return c.P.HasScope("inference") && c.Grants.Can(authz.Inference, project)
}

func (g *Gateway) prepare(r *http.Request, protocol, enginePath string) (*call, error) {
	caller, token, hint, ip, err := g.authenticate(r)
	if err != nil {
		return nil, err
	}
	raw, err := httpx.ReadBody(r, g.Config.MaxBodyBytes)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(r.Context(), db.CallTimeout)
	defer cancel()
	body, err := ParseBody(raw)
	if err != nil {
		return nil, httpx.Errorf(http.StatusBadRequest, "invalid_json", err.Error())
	}
	info, err := inspect(protocol, body, raw)
	if err != nil {
		return nil, err
	}
	model, grant, err := g.Catalog.Resolve(ctx, caller.P.ProjectID, info.Model)
	if errors.Is(err, catalog.ErrNotFound) || err == nil && !caller.AllowsModel(model.Name) && !caller.AllowsModel(info.Model) {
		return nil, notFound(info.Model)
	}
	if err != nil {
		return nil, err
	}
	if (protocol == providers.Embeddings) != (model.Kind == "embedding") {
		return nil, badParam("model", "model "+info.Model+" does not serve this endpoint")
	}
	if err := applyMax(body, &info, grant, g.Config.DefaultMaxTokens, protocol); err != nil {
		return nil, err
	}
	project, err := g.Policies.Project(ctx, caller.P.ProjectID)
	if err != nil {
		return nil, err
	}
	routes, err := g.Catalog.ProviderRoutes(ctx, model.ID)
	if err != nil {
		return nil, err
	}
	pl := decide(model, routes, protocol, classify(r, project, caller), g.Policies.Offline(ctx), caller)
	if pl, err = screen(pl, project, raw); err != nil {
		g.auditGuard(r, caller, model, pl.findings, "denied")
		return nil, err
	}
	if len(pl.findings) > 0 {
		g.auditGuard(r, caller, model, pl.findings, "kept_local")
	}
	c := &call{protocol: protocol, enginePath: enginePath, dialect: dialectFor(protocol), caller: caller, token: token, hint: hint, ip: ip,
		project: project, model: model, info: info, body: body, plan: pl, requestID: httpx.RequestID(ctx), client: clientLabel(r),
		headers: forwardHeaders(r)}
	if c.requestID == "" {
		c.requestID = httpx.NewRequestID()
	}
	if err := g.continuation(ctx, c); err != nil {
		return nil, err
	}
	if err := g.admissionRequest(ctx, r, c); err != nil {
		return nil, err
	}
	return c, nil
}

func notFound(name string) error {
	e := httpx.Errorf(http.StatusNotFound, "model_not_found", "The model `"+name+"` does not exist or you do not have access to it.")
	e.Param = "model"
	return e
}

// applyMax enforces the grant's output cap and fills a default when the client sent none.
func applyMax(b *Body, in *Info, grant catalog.Grant, def int, protocol string) error {
	if in.MaxField == "" {
		return nil
	}
	if in.HasMax {
		if grant.MaxOutputTokens > 0 && in.MaxTokens > int64(grant.MaxOutputTokens) {
			return badParam(in.MaxField, "exceeds this project's output limit of "+strconv.Itoa(grant.MaxOutputTokens)+" tokens")
		}
	} else {
		n := min(def, maxOutput)
		if grant.MaxOutputTokens > 0 && grant.MaxOutputTokens < n {
			n = grant.MaxOutputTokens
		}
		in.MaxTokens = int64(n)
		if err := b.Set(in.MaxField, n); err != nil {
			return err
		}
	}
	if grant.MaxContext > 0 && in.MaxTokens > int64(grant.MaxContext)-in.Estimate*3/4 {
		e := httpx.Errorf(http.StatusBadRequest, "context_length_exceeded", "This project's maximum context length is "+strconv.Itoa(grant.MaxContext)+" tokens.")
		e.Param = "messages"
		return e
	}
	return nil
}

func (g *Gateway) admissionRequest(ctx context.Context, r *http.Request, c *call) error {
	priority := prioritise(r, c.info, c.project, c.caller)
	out := c.info.MaxTokens * max(c.info.Choices, 1)
	tokens := c.info.Estimate + out
	micros := c.model.Chargeback.Cost(c.info.Estimate, out, 0)
	for _, rt := range c.plan.cloud {
		micros = max(micros, rt.Prices.Cost(c.info.Estimate, out, 0))
	}
	p := c.caller.P
	s := admission.Subject{ProjectID: p.ProjectID, UserID: p.UserID, KeyID: p.KeyID, ModelID: c.model.ID, TaskID: p.TaskID, TaskRootID: p.TaskRootID}
	if p.Type == "service_account" {
		s.ServiceAccountID = p.ID
	}
	if t := c.caller.Task; t != nil {
		s.TaskConcurrency, s.TreeConcurrency, s.TaskBudget = t.MaxConcurrency, t.TreeConcurrency, t.BudgetTokens
	}
	limits, err := g.loadLimits(ctx, s)
	if err != nil {
		return err
	}
	scopes, rates, budgets := admission.Build(limits, s, time.Now(), tokens, micros)
	wait := g.Config.QueueStandard.Duration
	switch priority {
	case levels.Interactive:
		wait = g.Config.QueueInteractive.Duration
	case levels.Batch:
		wait = g.Config.QueueBatch.Duration
	}
	var pins []authn.Pin
	if t := c.caller.Task; t != nil {
		pins = t.Pins[c.model.Name]
	}
	c.req = admission.Request{RequestID: c.requestID, ModelID: c.model.ID, Priority: priority, Class: c.plan.class, Local: c.plan.local,
		CloudFirst: c.plan.cloudFirst, Routes: c.plan.cloud, Pins: pins, Scopes: scopes, Rates: rates, Budgets: budgets, Tokens: tokens,
		Micros: micros, Affinity: c.info.Affinity, QueueUntil: time.Now().Add(wait), RunFor: g.Config.RequestTimeout.Duration, Only: c.req.Only,
		Isolated: c.project.Isolated,
		Who:      admission.Who{PrincipalType: string(p.Type), PrincipalID: p.ID, ProjectID: p.ProjectID, Client: c.client}}
	return nil
}

func (g *Gateway) loadLimits(ctx context.Context, s admission.Subject) ([]admission.Limit, error) {
	key := s.ProjectID + "|" + s.UserID + "|" + s.ServiceAccountID + "|" + s.KeyID + "|" + s.ModelID
	g.limitsMu.Lock()
	if c, ok := g.limits[key]; ok && time.Now().Before(c.expires) {
		g.limitsMu.Unlock()
		return c.l, nil
	}
	g.limitsMu.Unlock()
	l, err := admission.LoadLimits(ctx, g.Pool, s)
	if err != nil {
		return nil, err
	}
	g.limitsMu.Lock()
	if g.limits == nil || len(g.limits) > 50000 {
		g.limits = map[string]cachedLimits{}
	}
	g.limits[key] = cachedLimits{l, time.Now().Add(5 * time.Second)}
	g.limitsMu.Unlock()
	return l, nil
}

func (g *Gateway) auditGuard(r *http.Request, c *authn.Caller, m catalog.Model, fs []guard.Finding, outcome string) {
	e := audit.FromRequest(r, g.Proxies, c.P)
	e.Action, e.Outcome, e.TargetType, e.TargetID = "guard.credentials_detected", outcome, "model", m.Name
	e.Details = map[string]any{"kinds": guard.Kinds(fs), "project_id": c.P.ProjectID}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), db.CallTimeout)
	defer cancel()
	_ = audit.Log(ctx, g.Pool, e)
}

func clientLabel(r *http.Request) string {
	s := r.Header.Get("X-Starlight-Client")
	if s == "" {
		s = r.UserAgent()
	}
	s = strings.Map(func(c rune) rune {
		if c < 0x20 || c > 0x7e {
			return -1
		}
		return c
	}, s)
	if len(s) > 80 {
		s = s[:80]
	}
	return s
}

// forwardHeaders keeps only the protocol headers providers need from the client.
func forwardHeaders(r *http.Request) http.Header {
	h := http.Header{}
	for _, k := range []string{"Accept", "Anthropic-Version", "Anthropic-Beta"} {
		if v := r.Header.Get(k); v != "" {
			h.Set(k, v)
		}
	}
	return h
}
