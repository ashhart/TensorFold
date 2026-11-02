// Package bridge is Connect's loopback-only, app-key gated local endpoint; it calls Starlight as the signed-in user.
package bridge

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/appkeys"
	"github.com/ashhart/TensorFold/starlight/internal/connect/profile"
	"github.com/ashhart/TensorFold/starlight/internal/connect/starlightapi"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

// Config wires the endpoint to Starlight and to the local app registry.
type Config struct {
	Port           int
	Upstream       string
	HTTP           *http.Client
	Tokens         starlightapi.TokenSource
	Apps           *appkeys.Registry
	AllowedOrigins []string
	Project        string
	MaxReplayBody  int64
}

// Server is a running local endpoint.
type Server struct {
	cfg  Config
	base *url.URL
	srv  *http.Server
	ln   net.Listener
}

// New validates cfg; nothing listens until Start.
func New(cfg Config) (*Server, error) {
	base, err := url.Parse(cfg.Upstream)
	if err != nil || base.Host == "" {
		return nil, fmt.Errorf("bridge upstream %q is not absolute", cfg.Upstream)
	}
	if cfg.HTTP == nil || cfg.Tokens == nil || cfg.Apps == nil {
		return nil, errors.New("bridge needs an HTTP client, a token source and an app registry")
	}
	if cfg.MaxReplayBody <= 0 {
		cfg.MaxReplayBody = 32 << 20
	}
	s := &Server{cfg: cfg, base: base}
	s.srv = &http.Server{Handler: s.Handler(), ReadHeaderTimeout: 15 * time.Second, IdleTimeout: 120 * time.Second}
	return s, nil
}

// Start listens on 127.0.0.1 only and serves in the background.
func (s *Server) Start() error {
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", s.cfg.Port))
	if err != nil {
		return fmt.Errorf("local endpoint: %w", err)
	}
	s.ln = ln
	go s.srv.Serve(ln)
	return nil
}

// URL is the endpoint's base address, once started.
func (s *Server) URL() string {
	if s.ln == nil {
		return ""
	}
	return "http://" + s.ln.Addr().String()
}

// Shutdown stops accepting requests and waits for streams in flight until ctx ends.
func (s *Server) Shutdown(ctx context.Context) error { return s.srv.Shutdown(ctx) }

var (
	forwarded = map[string]bool{
		"GET /v1/models":                 true,
		"POST /v1/chat/completions":      true,
		"POST /v1/completions":           true,
		"POST /v1/embeddings":            true,
		"POST /v1/responses":             true,
		"POST /v1/messages":              true,
		"POST /v1/messages/count_tokens": true,
	}
	modelPath    = regexp.MustCompile(`^/v1/models/[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$`)
	responsePath = regexp.MustCompile(`^/v1/responses/[A-Za-z0-9_-]{1,128}$`)
)

// Allowed reports whether method and path may be forwarded to Starlight.
func Allowed(method, p string) bool {
	if path.Clean(p) != p || strings.Contains(p, "//") {
		return false
	}
	switch {
	case forwarded[method+" "+p]:
		return true
	case method == http.MethodGet && modelPath.MatchString(p):
		return true
	case (method == http.MethodGet || method == http.MethodDelete) && responsePath.MatchString(p):
		return true
	}
	return false
}

func dialect(p string) httpx.Dialect {
	if strings.HasPrefix(p, "/v1/messages") {
		return httpx.Anthropic
	}
	return httpx.OpenAI
}

// Handler applies the loopback, origin and app-key guards before forwarding.
func (s *Server) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		d := dialect(r.URL.Path)
		if !loopbackHost(r.Host) {
			httpx.Write(w, d, httpx.Errorf(http.StatusForbidden, "host_not_allowed", "the local endpoint answers only to 127.0.0.1 or localhost"))
			return
		}
		if o := r.Header.Get("Origin"); o != "" && !s.originAllowed(o) {
			httpx.Write(w, d, httpx.Errorf(http.StatusForbidden, "origin_not_allowed", "browser pages may not call the local endpoint"))
			return
		}
		if r.URL.Path == "/healthz" && r.Method == http.MethodGet {
			httpx.JSON(w, http.StatusOK, map[string]string{"status": "ok"})
			return
		}
		if !Allowed(r.Method, r.URL.Path) {
			httpx.Write(w, d, httpx.Errorf(http.StatusNotFound, "not_found", "the local endpoint does not serve this route"))
			return
		}
		app, ok := s.cfg.Apps.Lookup(httpx.BearerToken(r))
		if !ok {
			httpx.Write(w, d, httpx.Errorf(http.StatusUnauthorized, "invalid_api_key",
				"use an app key from `tensorfold-connect apps add NAME`; Starlight credentials are not accepted here"))
			return
		}
		s.forward(w, r, app, d)
	})
}

func loopbackHost(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	return profile.Loopback(host)
}

func (s *Server) originAllowed(origin string) bool {
	for _, o := range s.cfg.AllowedOrigins {
		if strings.EqualFold(strings.TrimRight(o, "/"), strings.TrimRight(origin, "/")) {
			return true
		}
	}
	return false
}
