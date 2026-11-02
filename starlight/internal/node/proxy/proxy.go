// Package proxy serves gateway traffic on a node: every request needs a gateway certificate and a valid permit.
package proxy

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
)

// DefaultMaxBody bounds a proxied request body.
const DefaultMaxBody = 64 << 20

// RefusedHeader marks answers given before the engine saw the request, so a gateway may retry elsewhere.
const RefusedHeader = "Starlight-Refused"

// Target is a replica the proxy can forward to.
type Target interface {
	Deployment() (string, int)
	Endpoint() (string, engine.Keys)
	Acquire() (func(), error)
}

// Backend finds replicas by id.
type Backend interface {
	Lookup(id string) (Target, bool)
}

// Supervised adapts an engine supervisor to Backend.
type Supervised struct{ S *engine.Supervisor }

// Lookup returns the supervisor's replica.
func (b Supervised) Lookup(id string) (Target, bool) {
	r, ok := b.S.Get(id)
	if !ok {
		return nil, false
	}
	return r, true
}

// Server verifies permits and streams engine replies back to the gateway.
type Server struct {
	NodeID    string
	Verifier  *permit.Verifier
	Replay    *permit.Replay
	Backend   Backend
	Transport http.RoundTripper
	Logger    *slog.Logger
	Now       func() time.Time
	MaxBody   int64
}

// NewTransport is a loopback-only transport that never consults proxy settings or rewrites encodings.
func NewTransport() *http.Transport {
	return &http.Transport{Proxy: nil, DisableCompression: true, MaxIdleConnsPerHost: 64, IdleConnTimeout: 90 * time.Second,
		ResponseHeaderTimeout: 0, ExpectContinueTimeout: 0}
}

// refuse answers without contacting the engine.
func refuse(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set(RefusedHeader, "1")
	e := httpx.Errorf(status, code, msg)
	if status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable {
		e.RetryAfter = 1
	}
	httpx.Write(w, httpx.OpenAI, e)
}

// ServeHTTP handles ReplicaPrefix + replica id + engine path; query strings are never forwarded.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
		refuse(w, http.StatusUnauthorized, "client_certificate_required", "a gateway certificate is required")
		return
	}
	rest, ok := strings.CutPrefix(r.URL.Path, wire.ReplicaPrefix)
	if !ok {
		refuse(w, http.StatusNotFound, "not_found", "unknown route")
		return
	}
	replicaID, enginePath, ok := strings.Cut(rest, "/")
	enginePath = "/" + enginePath
	if !ok || !ids.Valid(replicaID) || !wire.AllowedEngineRoute(r.Method, enginePath) {
		refuse(w, http.StatusNotFound, "route_not_allowed", "this engine route is not served through Starlight")
		return
	}
	token := r.Header.Get(wire.PermitHeader)
	if token == "" {
		refuse(w, http.StatusUnauthorized, "permit_required", "a Starlight permit is required")
		return
	}
	now := s.now()
	claims, err := s.Verifier.Verify(token, now)
	if err != nil {
		code := "permit_invalid"
		if errors.Is(err, permit.ErrExpired) {
			code = "permit_expired"
		}
		refuse(w, http.StatusForbidden, code, "the permit was not accepted")
		return
	}
	if claims.Node != s.NodeID || claims.Replica != replicaID || claims.Method != r.Method || claims.Path != enginePath {
		refuse(w, http.StatusForbidden, "permit_mismatch", "the permit names another node, replica or route")
		return
	}
	tgt, ok := s.Backend.Lookup(replicaID)
	if !ok {
		refuse(w, http.StatusServiceUnavailable, "replica_not_ready", "the replica is not on this node")
		return
	}
	if dep, rev := tgt.Deployment(); dep != claims.Deployment || rev != claims.Revision {
		refuse(w, http.StatusForbidden, "permit_mismatch", "the permit names another deployment revision")
		return
	}
	body, err := httpx.ReadBody(r, s.maxBody())
	if err != nil {
		w.Header().Set(RefusedHeader, "1")
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	sum := sha256.Sum256(body)
	if subtle.ConstantTimeCompare([]byte(hex.EncodeToString(sum[:])), []byte(claims.BodySHA256)) != 1 {
		refuse(w, http.StatusForbidden, "permit_mismatch", "the request body does not match the permit")
		return
	}
	if !s.Replay.Use(claims.RequestID, time.UnixMilli(claims.Expires), now) {
		refuse(w, http.StatusForbidden, "permit_replayed", "the permit was already used")
		return
	}
	release, err := tgt.Acquire()
	switch {
	case errors.Is(err, engine.ErrBusy):
		refuse(w, http.StatusTooManyRequests, "replica_busy", "every slot on this replica is in use")
		return
	case errors.Is(err, engine.ErrDraining):
		refuse(w, http.StatusServiceUnavailable, "replica_draining", "the replica is draining")
		return
	case err != nil:
		refuse(w, http.StatusServiceUnavailable, "replica_not_ready", "the replica is not ready")
		return
	}
	defer release()
	s.forward(w, r, tgt, enginePath, body, claims.RequestID)
}

func (s *Server) forward(w http.ResponseWriter, r *http.Request, tgt Target, target string, body []byte, rid string) {
	base, keys := tgt.Endpoint()
	if base == "" {
		refuse(w, http.StatusServiceUnavailable, "replica_not_ready", "the replica is not ready")
		return
	}
	up, err := http.NewRequestWithContext(r.Context(), r.Method, base+target, bytes.NewReader(body))
	if err != nil {
		refuse(w, http.StatusBadGateway, "engine_unavailable", "could not build the engine request")
		return
	}
	copyRequestHeaders(up.Header, r.Header)
	up.Header.Set("Authorization", "Bearer "+keyFor(r.Method, target, body, keys))
	up.ContentLength = int64(len(body))
	resp, err := s.transport().RoundTrip(up)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		s.logger().Warn("engine request failed", "request", rid, "error", err)
		var op *net.OpError
		if errors.As(err, &op) && op.Op == "dial" {
			refuse(w, http.StatusServiceUnavailable, "replica_not_ready", "the engine is not accepting connections")
			return
		}
		httpx.Write(w, httpx.OpenAI, httpx.Errorf(http.StatusBadGateway, "engine_unavailable", "the engine did not complete the request"))
		return
	}
	defer resp.Body.Close()
	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	stream(w, resp.Body)
}

// stream copies the body and flushes after every read so server-sent events arrive as they are produced.
func stream(w http.ResponseWriter, body io.Reader) {
	rc := http.NewResponseController(w)
	buf := make([]byte, 32<<10)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if rc.Flush() != nil {
				return
			}
		}
		if err != nil {
			return
		}
	}
}

func (s *Server) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

func (s *Server) maxBody() int64 {
	if s.MaxBody > 0 {
		return s.MaxBody
	}
	return DefaultMaxBody
}

func (s *Server) transport() http.RoundTripper {
	if s.Transport != nil {
		return s.Transport
	}
	return NewTransport()
}

func (s *Server) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}
