package oauth_test

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oauth"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

type pages struct {
	approve oauth.ApprovePage
	device  oauth.DevicePage
	message oauth.MessagePage
}

func (p *pages) Approve(w http.ResponseWriter, r *http.Request, a oauth.ApprovePage) { p.approve = a }
func (p *pages) Device(w http.ResponseWriter, r *http.Request, d oauth.DevicePage)   { p.device = d }
func (p *pages) Message(w http.ResponseWriter, r *http.Request, m oauth.MessagePage) {
	p.message = m
	w.WriteHeader(m.Status)
}

type rig struct {
	t      *testing.T
	srv    *httptest.Server
	pages  *pages
	cookie *http.Cookie
	csrf   string
	user   directory.User
	pool   *pgxpool.Pool
}

func newRig(t *testing.T) *rig {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "dee@example.com", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	sessions := session.New(pool, session.Config{})
	rec := httptest.NewRecorder()
	s, err := sessions.Create(ctx, rec, httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID})
	if err != nil {
		t.Fatal(err)
	}
	p := &pages{}
	proxies, _ := httpx.NewProxies(nil)
	mux := http.NewServeMux()
	issuer := &oauth.Issuer{Pool: pool, Life: oauth.Lifetimes{Access: 10 * time.Minute, RefreshIdle: time.Hour, RefreshAbsolute: 24 * time.Hour}}
	(&oauth.Server{Pool: pool, Issuer: issuer, Sessions: sessions, Pages: p, Proxies: proxies, PublicURL: "https://ai.example.com"}).Register(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return &rig{t: t, srv: srv, pages: p, cookie: rec.Result().Cookies()[0], csrf: s.CSRF(), user: u, pool: pool}
}

func noRedirect() *http.Client {
	return &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}

func (r *rig) browser(method, path string, form url.Values) *http.Response {
	var body *strings.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	} else {
		body = strings.NewReader("")
	}
	req, _ := http.NewRequest(method, r.srv.URL+path, body)
	req.AddCookie(r.cookie)
	if form != nil {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set("Origin", r.srv.URL)
	}
	resp, err := noRedirect().Do(req)
	if err != nil {
		r.t.Fatal(err)
	}
	resp.Body.Close()
	return resp
}

func (r *rig) tokenCall(form url.Values) (int, map[string]any) {
	resp, err := http.PostForm(r.srv.URL+"/oauth/token", form)
	if err != nil {
		r.t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func pkce() (verifier, challenge string) {
	verifier = strings.Repeat("v", 43) + "abc"
	sum := sha256.Sum256([]byte(verifier))
	return verifier, base64.RawURLEncoding.EncodeToString(sum[:])
}

func TestPKCEFlowRotationAndReuseDetection(t *testing.T) {
	r := newRig(t)
	verifier, challenge := pkce()
	q := url.Values{"client_id": {discovery.ClientConnect}, "redirect_uri": {"http://127.0.0.1:53111/callback"}, "response_type": {"code"},
		"code_challenge": {challenge}, "code_challenge_method": {"S256"}, "state": {"state-123456"}, "scope": {"inference offline_access"},
		"device_name": {"laptop\x07"}}
	if resp := r.browser("GET", "/oauth/authorize?"+q.Encode(), nil); resp.StatusCode != 200 || r.pages.approve.DeviceName != "laptop" {
		t.Fatalf("authorize form %d %+v", resp.StatusCode, r.pages.approve)
	}
	form := url.Values{}
	for k, v := range r.pages.approve.Hidden {
		form.Set(k, v)
	}
	form.Set("decision", "allow")
	form.Set("_csrf", r.csrf)
	resp := r.browser("POST", "/oauth/authorize", form)
	loc, _ := url.Parse(resp.Header.Get("Location"))
	if resp.StatusCode != http.StatusSeeOther || loc.Host != "127.0.0.1:53111" || loc.Query().Get("state") != "state-123456" {
		t.Fatalf("redirect %d %s", resp.StatusCode, loc)
	}
	code := loc.Query().Get("code")
	exchange := url.Values{"grant_type": {"authorization_code"}, "client_id": {discovery.ClientConnect}, "code": {code},
		"redirect_uri": {"http://127.0.0.1:53111/callback"}, "code_verifier": {verifier}}
	bad := url.Values{}
	for k, v := range exchange {
		bad[k] = v
	}
	bad.Set("code_verifier", strings.Repeat("x", 43))
	if st, _ := r.tokenCall(bad); st != 400 {
		t.Fatalf("wrong verifier accepted: %d", st)
	}
	if st, _ := r.tokenCall(exchange); st != 400 {
		t.Fatal("code survived a failed exchange; codes must be single-use")
	}
	r.browser("POST", "/oauth/authorize", form)
	resp = r.browser("POST", "/oauth/authorize", form)
	loc, _ = url.Parse(resp.Header.Get("Location"))
	exchange.Set("code", loc.Query().Get("code"))
	st, tok := r.tokenCall(exchange)
	if st != 200 || !strings.HasPrefix(tok["access_token"].(string), "sla_") || !strings.HasPrefix(tok["refresh_token"].(string), "slf_") {
		t.Fatalf("exchange = %d %v", st, tok)
	}
	refresh := url.Values{"grant_type": {"refresh_token"}, "client_id": {discovery.ClientConnect}, "refresh_token": {tok["refresh_token"].(string)}}
	st, next := r.tokenCall(refresh)
	if st != 200 || next["refresh_token"] == tok["refresh_token"] {
		t.Fatalf("rotation = %d %v", st, next)
	}
	if st, _ := r.tokenCall(refresh); st != 400 {
		t.Fatal("reused refresh token accepted")
	}
	refresh.Set("refresh_token", next["refresh_token"].(string))
	if st, _ := r.tokenCall(refresh); st != 400 {
		t.Fatal("refresh family survived reuse detection")
	}
}

func TestDeviceGrant(t *testing.T) {
	r := newRig(t)
	resp, err := http.PostForm(r.srv.URL+"/oauth/device/code", url.Values{"client_id": {discovery.ClientCLI}, "scope": {"inference"}})
	if err != nil {
		t.Fatal(err)
	}
	var dr discovery.DeviceResponse
	_ = json.NewDecoder(resp.Body).Decode(&dr)
	resp.Body.Close()
	if dr.Interval != 5 || !strings.HasSuffix(dr.VerificationURI, "/device") || len(dr.UserCode) != 9 {
		t.Fatalf("device response %+v", dr)
	}
	poll := url.Values{"grant_type": {"urn:ietf:params:oauth:grant-type:device_code"}, "client_id": {discovery.ClientCLI}, "device_code": {dr.DeviceCode}}
	if _, out := r.tokenCall(poll); out["error"] != "authorization_pending" {
		t.Fatalf("first poll %v", out)
	}
	if _, out := r.tokenCall(poll); out["error"] != "slow_down" {
		t.Fatalf("fast poll %v", out)
	}
	r.browser("POST", "/device", url.Values{"user_code": {strings.ToLower(dr.UserCode)}, "_csrf": {r.csrf}})
	if !r.pages.device.Confirm || r.pages.device.ClientName != "TensorFold CLI" {
		t.Fatalf("confirm page %+v", r.pages.device)
	}
	r.browser("POST", "/device/decide", url.Values{"user_code": {dr.UserCode}, "decision": {"allow"}, "_csrf": {r.csrf}})
	if r.pages.message.Status != 200 {
		t.Fatalf("decide %+v", r.pages.message)
	}
	r.rewind()
	st, out := r.tokenCall(poll)
	if st != 200 || out["refresh_token"] != nil {
		t.Fatalf("approved poll = %d %v", st, out)
	}
	r.rewind()
	if _, out := r.tokenCall(poll); out["error"] != "invalid_grant" {
		t.Fatalf("consumed code reused: %v", out)
	}
}

func TestRedirectAndScopeRules(t *testing.T) {
	c, _ := oauth.LookupClient(discovery.ClientConnect)
	for raw, want := range map[string]bool{
		"http://127.0.0.1:5000/cb": true, "http://[::1]:5000/cb": true, "http://localhost:5000/cb": false,
		"https://127.0.0.1:5000/cb": false, "http://127.0.0.1/cb": false, "http://198.51.100.1:5000/cb": false, "http://127.0.0.1:5000/cb?x=1": false,
	} {
		if c.ValidRedirect(raw) != want {
			t.Errorf("ValidRedirect(%q) != %v", raw, want)
		}
	}
	if _, ok := c.NarrowScope("inference admin"); ok {
		t.Fatal("granted an unknown scope")
	}
	cli, _ := oauth.LookupClient(discovery.ClientCLI)
	if cli.ValidRedirect("http://127.0.0.1:5000/cb") {
		t.Fatal("CLI client must not use redirects")
	}
}

func (r *rig) rewind() {
	if _, err := r.pool.Exec(context.Background(), "UPDATE device_grants SET last_polled_at = now() - interval '1 hour'"); err != nil {
		r.t.Fatal(err)
	}
}
