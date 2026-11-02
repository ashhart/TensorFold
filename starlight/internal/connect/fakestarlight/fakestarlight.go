// Package fakestarlight is an in-process stand-in for Starlight's OAuth and API endpoints, used by Connect's tests.
package fakestarlight

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/connect/profile"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

type code struct {
	challenge, redirect string
}

type refresh struct {
	family string
	used   bool
}

// Request is one API call the fake received.
type Request struct {
	Method, Path string
	Header       http.Header
	Body         string
}

// Starlight records what Connect sent and answers like Starlight does.
type Starlight struct {
	*httptest.Server
	Me            discovery.Me
	Models        []string
	DevicePending int
	DeviceSlow    bool
	AccessTTL     int
	RefreshCalls  atomic.Int32
	Gate          chan struct{}
	mu            sync.Mutex
	codes         map[string]code
	refresh       map[string]*refresh
	access        map[string]string
	revokedFamily map[string]bool
	requests      []Request
}

// New starts a fake Starlight on a loopback port.
func New(t testing.TB) *Starlight {
	f := &Starlight{codes: map[string]code{}, refresh: map[string]*refresh{}, access: map[string]string{}, revokedFamily: map[string]bool{},
		AccessTTL: 600, Models: []string{"qwen3.8-27b", "nemotron-lightning"},
		Me: discovery.Me{UserID: "u-1", Email: "ada@example.com", DisplayName: "Ada Example", OrgName: "Example Org",
			Projects: []discovery.Project{{ID: "p-1", Slug: "research", Name: "Research", Role: "member", Classification: "internal", Default: true}}}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET "+discovery.Path, f.discovery)
	mux.HandleFunc("GET /oauth/authorize", f.authorize)
	mux.HandleFunc("POST /oauth/token", f.token)
	mux.HandleFunc("POST /oauth/device/code", f.device)
	mux.HandleFunc("POST /oauth/revoke", f.revoke)
	mux.HandleFunc("GET /oauth/userinfo", f.authed(f.userinfo))
	mux.HandleFunc("GET /v1/models", f.authed(f.models))
	mux.HandleFunc("/v1/", f.authed(f.inference))
	f.Server = httptest.NewServer(mux)
	t.Cleanup(f.Close)
	return f
}

// Doc is the discovery document this fake serves.
func (f *Starlight) Doc() discovery.Document {
	return discovery.Build(f.URL, f.Me.OrgName, "test", "0190f5a0-0000-7000-8000-000000000001", false)
}

// Issue mints a token pair in a fresh family, as a completed sign-in would.
func (f *Starlight) Issue() discovery.TokenResponse {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.issueLocked(secret.RandomString(8))
}

func (f *Starlight) issueLocked(family string) discovery.TokenResponse {
	at, _ := secret.Generate(secret.Access)
	rt, _ := secret.Generate(secret.Refresh)
	f.access[at] = family
	f.refresh[rt] = &refresh{family: family}
	return discovery.TokenResponse{AccessToken: at, TokenType: "Bearer", ExpiresIn: f.AccessTTL, RefreshToken: rt, Scope: "inference offline_access profile"}
}

// Expire makes an access token fail, as an expiry or revocation would.
func (f *Starlight) Expire(accessToken string) {
	f.mu.Lock()
	delete(f.access, accessToken)
	f.mu.Unlock()
}

// Requests returns the API calls received so far.
func (f *Starlight) Requests() []Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]Request(nil), f.requests...)
}

func (f *Starlight) discovery(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, f.Doc()) }

func (f *Starlight) authorize(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	redirect, err := url.Parse(q.Get("redirect_uri"))
	if q.Get("client_id") != discovery.ClientConnect || q.Get("response_type") != "code" || q.Get("code_challenge_method") != "S256" ||
		err != nil || redirect.Scheme != "http" || !profile.Loopback(redirect.Hostname()) || q.Get("state") == "" {
		http.Error(w, "bad authorization request", http.StatusBadRequest)
		return
	}
	c := secret.RandomString(16)
	f.mu.Lock()
	f.codes[c] = code{challenge: q.Get("code_challenge"), redirect: redirect.String()}
	f.mu.Unlock()
	back := redirect.Query()
	back.Set("code", c)
	back.Set("state", q.Get("state"))
	redirect.RawQuery = back.Encode()
	http.Redirect(w, r, redirect.String(), http.StatusFound)
}

func oauthError(w http.ResponseWriter, code string) {
	writeJSON(w, http.StatusBadRequest, map[string]string{"error": code})
}

func (f *Starlight) token(w http.ResponseWriter, r *http.Request) {
	if r.ParseForm() != nil || r.PostForm.Get("client_id") != discovery.ClientConnect {
		oauthError(w, "invalid_client")
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.PostForm.Get("grant_type") {
	case "authorization_code":
		c, ok := f.codes[r.PostForm.Get("code")]
		delete(f.codes, r.PostForm.Get("code"))
		sum := sha256.Sum256([]byte(r.PostForm.Get("code_verifier")))
		if !ok || c.redirect != r.PostForm.Get("redirect_uri") || base64.RawURLEncoding.EncodeToString(sum[:]) != c.challenge {
			oauthError(w, "invalid_grant")
			return
		}
		writeJSON(w, 200, f.issueLocked(secret.RandomString(8)))
	case "refresh_token":
		f.RefreshCalls.Add(1)
		rt := f.refresh[r.PostForm.Get("refresh_token")]
		if rt == nil || f.revokedFamily[rt.family] {
			oauthError(w, "invalid_grant")
			return
		}
		if rt.used {
			f.revokedFamily[rt.family] = true
			oauthError(w, "invalid_grant")
			return
		}
		rt.used = true
		writeJSON(w, 200, f.issueLocked(rt.family))
	case "urn:ietf:params:oauth:grant-type:device_code":
		if r.PostForm.Get("device_code") != "device-code-1" {
			oauthError(w, "invalid_grant")
			return
		}
		if f.DeviceSlow {
			f.DeviceSlow = false
			oauthError(w, "slow_down")
			return
		}
		if f.DevicePending > 0 {
			f.DevicePending--
			oauthError(w, "authorization_pending")
			return
		}
		writeJSON(w, 200, f.issueLocked(secret.RandomString(8)))
	default:
		oauthError(w, "unsupported_grant_type")
	}
}

func (f *Starlight) device(w http.ResponseWriter, r *http.Request) {
	if r.ParseForm() != nil || r.PostForm.Get("client_id") != discovery.ClientConnect {
		oauthError(w, "invalid_client")
		return
	}
	writeJSON(w, 200, discovery.DeviceResponse{DeviceCode: "device-code-1", UserCode: "BCDF-GHJK", VerificationURI: f.URL + "/device",
		VerificationURIComplete: f.URL + "/device?user_code=BCDF-GHJK", ExpiresIn: 600, Interval: 1})
}

func (f *Starlight) revoke(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	f.mu.Lock()
	if rt := f.refresh[r.PostForm.Get("token")]; rt != nil {
		f.revokedFamily[rt.family] = true
	}
	f.mu.Unlock()
	w.WriteHeader(http.StatusOK)
}

func (f *Starlight) authed(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		tok := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		f.mu.Lock()
		family, ok := f.access[tok]
		ok = ok && !f.revokedFamily[family]
		f.mu.Unlock()
		if !ok {
			writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]any{"message": "invalid token", "type": "authentication_error"}})
			return
		}
		next(w, r)
	}
}

func (f *Starlight) userinfo(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, f.Me) }

func (f *Starlight) models(w http.ResponseWriter, r *http.Request) {
	data := []map[string]any{}
	for _, m := range f.Models {
		data = append(data, map[string]any{"id": m, "object": "model", "owned_by": "starlight"})
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func (f *Starlight) inference(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	f.mu.Lock()
	f.requests = append(f.requests, Request{Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Body: string(body)})
	gate := f.Gate
	f.mu.Unlock()
	if !strings.Contains(string(body), `"stream":true`) {
		writeJSON(w, 200, map[string]any{"id": "chatcmpl-1", "object": "chat.completion", "choices": []any{}})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	flusher := w.(http.Flusher)
	for i := 0; i < 3; i++ {
		fmt.Fprintf(w, "data: {\"chunk\":%d}\n\n", i)
		flusher.Flush()
		select {
		case <-r.Context().Done():
			return
		case <-release(gate):
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func release(gate chan struct{}) <-chan struct{} {
	if gate == nil {
		ch := make(chan struct{})
		close(ch)
		return ch
	}
	return gate
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
