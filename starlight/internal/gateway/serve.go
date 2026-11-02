package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptrace"
	"sync/atomic"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

var (
	errRevoked = errors.New("credential revoked during the request")
	errTooLong = errors.New("request exceeded the gateway's maximum duration")
)

// serve admits, dispatches with failover on proven refusals, relays and settles.
func (g *Gateway) serve(w http.ResponseWriter, r *http.Request, c *call, start time.Time) {
	limit := g.Config.RequestTimeout.Duration
	if limit <= 0 {
		limit = 30 * time.Minute
	}
	bounded, stopTimer := context.WithTimeoutCause(r.Context(), limit, errTooLong)
	defer stopTimer()
	ctx, cancel := context.WithCancelCause(bounded)
	defer cancel(nil)
	for attempt := 0; attempt < 3; attempt++ {
		lease, err := g.Admitter.Admit(ctx, c.req)
		if err != nil {
			g.refuse(w, c.dialect, err)
			return
		}
		resp, sent, err := g.dispatch(ctx, c, lease)
		if err != nil && !sent || refused(resp) {
			if resp != nil {
				resp.Body.Close()
			}
			g.settle(ctx, c, lease, admission.Usage{Observed: true}, 0, "not_dispatched", 0, start)
			c.req.Exclude, c.req.Retry = append(c.req.Exclude, targetID(lease.Target)), true
			continue
		}
		if err != nil {
			g.settle(ctx, c, lease, admission.Usage{}, http.StatusBadGateway, "upstream_unreachable", 0, start)
			g.refuse(w, c.dialect, httpx.Errorf(http.StatusBadGateway, "upstream_unreachable", "the model server did not answer"))
			return
		}
		stop := g.watch(ctx, cancel, c)
		route := "local"
		if !lease.Target.Local() {
			route = "cloud"
		}
		if g.Metrics != nil {
			g.Metrics.InFlight.WithLabelValues(route).Inc()
			defer g.Metrics.InFlight.WithLabelValues(route).Dec()
		}
		w.Header().Set("X-Starlight-Route", route)
		w.Header().Set("X-Starlight-Model", c.model.Name)
		strip := c.protocol == providers.ChatCompletions && c.info.Stream && !c.info.WantsUsage
		out := relay(ctx, w, resp, c.protocol, strip, c.dialect, start)
		resp.Body.Close()
		stop()
		switch cause := context.Cause(ctx); {
		case errors.Is(cause, errRevoked):
			out.outcome = "revoked"
		case errors.Is(cause, errTooLong):
			out.outcome = "timeout"
		}
		g.settle(ctx, c, lease, out.usage, out.status, out.outcome, out.firstByte, start)
		g.remember(ctx, c, lease, out)
		return
	}
	g.refuse(w, c.dialect, admission.ErrCapacity)
}

// engineOnly are TensorFold engine extensions that cloud APIs reject; cloud routes never receive them.
var engineOnly = []string{"priority", "draft", "thinking_budget", "grammar_budget", "return_token_ids", "chat_template_kwargs", "guided_json",
	"guided_regex", "guided_choice", "guided_grammar", "structured_outputs", "ignore_eos", "min_p", "top_k"}

func targetID(t admission.Target) string {
	if t.Local() {
		return t.ReplicaID
	}
	return t.Route.ID
}

// dispatch rewrites the body for the chosen target and sends it.
func (g *Gateway) dispatch(ctx context.Context, c *call, l *admission.Lease) (*http.Response, bool, error) {
	b := c.body
	if l.Target.Local() {
		_ = b.Set("model", l.Target.ServedName)
		if c.req.Priority == levels.Batch {
			_ = b.Set("priority", "background")
		}
	} else {
		_ = b.Set("model", l.Target.Route.UpstreamModel)
		for _, f := range engineOnly {
			if f != "top_k" || l.Target.Route.Kind != "anthropic" {
				b.Delete(f)
			}
		}
	}
	if c.protocol == providers.ChatCompletions && c.info.Stream {
		_ = b.Set("stream_options", map[string]bool{"include_usage": true})
	}
	body := b.Bytes()
	if l.Target.Local() {
		d, err := g.Nodes.Do(ctx, l, http.MethodPost, c.enginePath, body, subject(c.caller), c.req.Priority.String(), int(c.info.MaxTokens))
		return d.Resp, d.Sent, err
	}
	var sent atomic.Bool
	tctx := httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{WroteRequest: func(httptrace.WroteRequestInfo) { sent.Store(true) }})
	resp, err := g.Providers.Do(tctx, *l.Target.Route, c.protocol, body, c.headers)
	return resp, sent.Load() || resp != nil, err
}

// subject is a stable pseudonym for the caller, so node logs never carry user ids.
func subject(c *authn.Caller) string {
	sum := sha256.Sum256([]byte("starlight-subject:" + string(c.P.Type) + ":" + c.P.ID))
	return hex.EncodeToString(sum[:8])
}

// watch re-checks the credential while a request runs and cancels it when the credential is revoked.
func (g *Gateway) watch(ctx context.Context, cancel context.CancelCauseFunc, c *call) func() {
	done := make(chan struct{})
	go func() {
		t := time.NewTicker(10 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-t.C:
				if g.revoked(ctx, c) {
					cancel(errRevoked)
					return
				}
			}
		}
	}()
	return func() { close(done) }
}

// revoked repeats authenticate's checks: the credential, its project, and its scope and grant in the request's project.
func (g *Gateway) revoked(ctx context.Context, c *call) bool {
	caller, err := g.Auth.Authenticate(ctx, c.token, c.hint, c.ip)
	if errors.Is(err, authn.ErrInvalid) || errors.Is(err, authn.ErrProject) {
		return true
	}
	return err == nil && !mayInfer(caller, c.caller.P.ProjectID)
}

func (g *Gateway) settle(ctx context.Context, c *call, l *admission.Lease, u admission.Usage, status int, outcome string, first time.Duration, start time.Time) {
	route, price := "local", c.model.Chargeback
	if !l.Target.Local() {
		route, price = "cloud", l.Target.Route.Prices
	}
	if u.Observed {
		u.Micros = price.Cost(u.Prompt, u.Completion, u.Cached)
	}
	p := c.caller.P
	rec := admission.Record{RequestID: c.requestID, PrincipalType: string(p.Type), PrincipalID: p.ID, UserID: p.UserID, ProjectID: p.ProjectID,
		APIKeyID: p.KeyID, TaskID: p.TaskID, ModelName: c.model.Name, Protocol: c.protocol, Status: status, Outcome: outcome, Queue: l.Queued,
		TTFT: first, Duration: time.Since(start), Classification: c.plan.class.String(), Client: c.client}
	if err := g.Admitter.Settle(ctx, l, u, rec); err != nil && g.Log != nil {
		g.Log.Error("settle failed", "request_id", c.requestID, "error", err)
	}
	if g.Metrics != nil {
		g.Metrics.Observe(c.model.Name, c.protocol, route, status, outcome, time.Since(start), first, u.Prompt, u.Completion, u.Cached)
		if l.Queued > 0 {
			g.Metrics.QueueWait.WithLabelValues(c.model.Name, c.req.Priority.String()).Observe(l.Queued.Seconds())
		}
	}
}

// refuse maps policy and admission errors to client errors without revealing internals.
func (g *Gateway) refuse(w http.ResponseWriter, d httpx.Dialect, err error) {
	var le *admission.LimitError
	var he *httpx.Error
	switch {
	case errors.As(err, &he):
		g.count(he.Code)
		httpx.Write(w, d, he)
	case errors.Is(err, authn.ErrInvalid):
		g.count("unauthenticated")
		httpx.Write(w, d, httpx.ErrUnauthenticated)
	case errors.Is(err, authn.ErrProject):
		g.count("project_required")
		httpx.Write(w, d, httpx.Errorf(http.StatusForbidden, "project_required", "choose a project you belong to with the X-Starlight-Project header"))
	case errors.As(err, &le):
		g.count(le.Kind + "_limit")
		e := httpx.Errorf(http.StatusTooManyRequests, le.Kind+"_limit", le.Kind+" limit reached; retry later")
		e.RetryAfter = le.RetryAfter
		httpx.Write(w, d, e)
	case errors.Is(err, admission.ErrCapacity):
		g.count("capacity")
		e := httpx.Errorf(http.StatusServiceUnavailable, "capacity_unavailable", "every eligible model server is busy; retry shortly")
		e.RetryAfter = 2
		httpx.Write(w, d, e)
	case errors.Is(err, admission.ErrNoRoute):
		g.count("no_route")
		httpx.Write(w, d, httpx.Errorf(http.StatusServiceUnavailable, "no_route", "no deployment or provider may serve this request under its data policy"))
	case errors.Is(err, context.Canceled):
		g.count("client_left")
	case db.Unavailable(err):
		g.count("database_unavailable")
		e := httpx.Errorf(http.StatusServiceUnavailable, "temporarily_unavailable", "Starlight cannot reach its database or identity provider; retry shortly")
		e.RetryAfter = 1
		httpx.Write(w, d, e)
	default:
		g.count("internal")
		if g.Log != nil {
			g.Log.Error("gateway error", "error", err)
		}
		httpx.Write(w, d, err)
	}
}

func (g *Gateway) count(reason string) {
	if g.Metrics != nil {
		g.Metrics.Rejections.WithLabelValues(reason).Inc()
	}
}
