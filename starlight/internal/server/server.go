package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/buildinfo"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/console/fleetpages"
	"github.com/ashhart/TensorFold/starlight/internal/console/identitypages"
	"github.com/ashhart/TensorFold/starlight/internal/control"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/gateway"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/login"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oauth"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/scim"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/metrics"
	"github.com/ashhart/TensorFold/starlight/internal/plans"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
	"github.com/ashhart/TensorFold/starlight/internal/tasks"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// Starlight is one running instance and every component it hosts.
type Starlight struct {
	Config   config.Config
	Log      *slog.Logger
	Pool     *pgxpool.Pool
	Keys     *keyring.Keyring
	Inst     Installation
	Trust    *trust.Trust
	Metrics  *metrics.Set
	Auth     *authn.Authenticator
	Catalog  *catalog.Catalog
	Admitter *admission.Admitter
	Policies *gateway.Policies
	Egress   *providers.Egress
	Sessions *session.Manager
	Pages    *web.Renderer
	Gateway  *gateway.Gateway
	Control  *control.Server
	Tasks    *tasks.Service
	Plans    *plans.Service
	Login    *login.Handler
	Console  *console.Console
	SCIM     *scim.Handler
	Instance string
	rateKey  []byte
	mount    []func(*http.ServeMux)
	orgMu    sync.Mutex
	orgAt    time.Time

	publicCert  *certFile
	credentials *providers.Credentials
}

// Open connects, checks the schema and keys, and builds every component without starting listeners.
func Open(ctx context.Context, cfg config.Config, log *slog.Logger) (*Starlight, error) {
	url, err := cfg.DatabaseURL()
	if err != nil {
		return nil, err
	}
	pool, err := db.Open(ctx, url, db.Options{MaxConns: cfg.Database.MaxConns, MinConns: cfg.Database.MinConns})
	if err != nil {
		return nil, err
	}
	if n, err := db.Pending(ctx, pool); err != nil || n > 0 {
		pool.Close()
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("database schema is %d migrations behind this build; run `starlight migrate`", n)
	}
	keys, err := LoadKeyring(ctx, pool, cfg.Keys)
	if err != nil {
		pool.Close()
		return nil, err
	}
	inst, err := LoadInstallation(ctx, pool)
	if err != nil {
		pool.Close()
		return nil, err
	}
	// Every instance derives the same rate-limit key from the master key, so their shared counters match.
	t := &Starlight{Config: cfg, Log: log, Pool: pool, Keys: keys, Inst: inst, Instance: ids.New(), Metrics: metrics.New(),
		rateKey: keys.Derive("starlight rate-limit counters", 32)}
	if t.Trust, err = trust.Load(ctx, pool, keys, t.Instance); err != nil {
		pool.Close()
		return nil, err
	}
	if err := t.build(); err != nil {
		pool.Close()
		return nil, err
	}
	return t, nil
}

// LoadKeyring unwraps data keys with the configured master key and any previous ones.
func LoadKeyring(ctx context.Context, q db.Querier, k config.Keys) (*keyring.Keyring, error) {
	master, err := keyring.LoadMaster(k.MasterKey, k.MasterKeyFile)
	if err != nil {
		return nil, err
	}
	var prev []*keyring.Master
	for _, f := range k.PreviousKeyFiles {
		m, err := keyring.LoadMaster("", f)
		if err != nil {
			return nil, err
		}
		prev = append(prev, m)
	}
	keys := keyring.New(master, prev...)
	return keys, keys.Load(ctx, q)
}

// org is the organisation name, re-read at most every five seconds so a rename shows on every instance.
func (t *Starlight) org(ctx context.Context) string {
	t.orgMu.Lock()
	defer t.orgMu.Unlock()
	if time.Since(t.orgAt) > 5*time.Second {
		var name string
		if err := t.Pool.QueryRow(ctx, "SELECT org_name FROM settings WHERE id").Scan(&name); err == nil && name != "" {
			t.Inst.OrgName = name
		}
		t.orgAt = time.Now()
	}
	return t.Inst.OrgName
}

func (t *Starlight) build() error {
	cfg := t.Config
	proxies, err := httpx.NewProxies(cfg.TrustedProxies)
	if err != nil {
		return err
	}
	t.Auth = &authn.Authenticator{Pool: t.Pool, TTL: cfg.Sessions.AuthCache.Duration}
	t.Catalog = &catalog.Catalog{Pool: t.Pool}
	t.Admitter = &admission.Admitter{Pool: t.Pool, GatewayID: t.Instance, Ageing: cfg.Gateway.AgeingStep.Duration, Waker: admission.NewWaker()}
	t.Policies = &gateway.Policies{Pool: t.Pool}
	t.Egress = providers.NewEgress()
	creds := &providers.Credentials{Pool: t.Pool, Keys: t.Keys}
	t.credentials = creds
	nodes := &gateway.Nodes{TrustDomain: t.Inst.TrustDomain, Roots: t.Trust.Roots(), Cert: t.Trust.GatewayCert, Signer: t.Trust.Signer}
	t.Gateway = &gateway.Gateway{Pool: t.Pool, Auth: t.Auth, Catalog: t.Catalog, Admitter: t.Admitter, Policies: t.Policies, Nodes: nodes,
		Providers: &providers.Client{Egress: t.Egress, Creds: creds}, Metrics: t.Metrics, Proxies: proxies, Config: cfg.Gateway, Log: t.Log}
	t.Control = &control.Server{Pool: t.Pool, Trust: t.Trust, Proxies: proxies, Log: t.Log, RateKey: t.rateKey}
	t.Sessions = session.New(t.Pool, session.Config{Idle: cfg.Sessions.ConsoleIdle.Duration, Absolute: cfg.Sessions.ConsoleAbsolute.Duration,
		Secure: cfg.Secure()})
	if t.Pages, err = web.New(t.org); err != nil {
		return err
	}
	t.Tasks = &tasks.Service{Pool: t.Pool, Auth: t.Auth, Catalog: t.Catalog, Proxies: proxies}
	chat := http.NewServeMux()
	t.Gateway.Register(chat)
	t.Plans = &plans.Service{Pool: t.Pool, Keys: t.Keys, Auth: t.Auth, Catalog: t.Catalog, Tasks: t.Tasks, Proxies: proxies, Chat: chat}
	idp, err := IDPClient(cfg.Identity.CAFile)
	if err != nil {
		return err
	}
	if t.Login, err = login.New(login.Config{PublicURL: cfg.PublicURL, Pool: t.Pool, Keyring: t.Keys, Sessions: t.Sessions, Proxies: proxies,
		Web: t.Pages, HTTP: idp, RateKey: t.rateKey}); err != nil {
		return err
	}
	t.Auth.JWT = func(ctx context.Context, raw string) (string, error) {
		if !oidc.LooksLikeJWT(raw) {
			return "", authn.ErrInvalid
		}
		u, _, err := t.Login.BearerUser(ctx, raw)
		return u.ID, err
	}
	t.SCIM = scim.New(scim.Options{Pool: t.Pool, BaseURL: cfg.PublicURL, Proxies: proxies, RateKey: t.rateKey})
	t.Console, err = console.New(&console.Console{Pool: t.Pool, Keys: t.Keys, Sessions: t.Sessions, Login: t.Login, Catalog: t.Catalog,
		Trust: t.Trust, Proxies: proxies, PublicURL: cfg.PublicURL, NodeURL: cfg.NodeURL, Version: version(), Org: t.org, Flush: t.flush},
		identitypages.Templates, fleetpages.Templates)
	if err != nil {
		return err
	}
	identitypages.Register(t.Console)
	fleetpages.Register(t.Console, fleetpages.WithPlans(t.Plans))
	return nil
}

// flush drops this instance's caches after an administrative edit; other instances converge within their TTLs.
func (t *Starlight) flush() {
	t.Catalog.Flush()
	t.Policies.Flush()
	t.Auth.Forget()
	t.credentials.Flush()
}

// IDPClient reaches identity providers, trusting the system roots plus an optional private CA bundle.
func IDPClient(caFile string) (*http.Client, error) {
	c := &http.Client{Timeout: 20 * time.Second}
	if caFile == "" {
		return c, nil
	}
	pem, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("identity.ca_file: %w", err)
	}
	roots, err := x509.SystemCertPool()
	if err != nil {
		roots = x509.NewCertPool()
	}
	if !roots.AppendCertsFromPEM(pem) {
		return nil, errors.New("identity.ca_file holds no PEM certificates")
	}
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12, RootCAs: roots}
	c.Transport = t
	return c, nil
}

// Mount adds routes to the public listener, for components assembled outside this package.
func (t *Starlight) Mount(f func(*http.ServeMux)) { t.mount = append(t.mount, f) }

// Handler is the public listener's routes for this instance's roles.
func (t *Starlight) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /readyz", t.ready)
	mux.HandleFunc("GET "+discoveryPath, t.discovery)
	if t.Config.HasRole("gateway") {
		t.Gateway.Register(mux)
		t.Tasks.Register(mux)
		t.Plans.Register(mux)
	}
	if t.Config.HasRole("console") {
		issuer := &oauth.Issuer{Pool: t.Pool, Life: oauth.Lifetimes{Access: t.Config.Sessions.AccessTTL.Duration,
			RefreshIdle: t.Config.Sessions.RefreshIdle.Duration, RefreshAbsolute: t.Config.Sessions.RefreshAbsolute.Duration}}
		(&oauth.Server{Pool: t.Pool, Issuer: issuer, Sessions: t.Sessions, Pages: oauthPages{t.Pages}, Proxies: t.Gateway.Proxies,
			PublicURL: t.Config.PublicURL, OrgName: t.org, Me: meBuilder(t.Pool, t.Catalog, t.org), AccessFor: t.accessOwner,
			RateKey: t.rateKey}).Register(mux)
		mux.Handle("GET /static/", web.StaticHandler())
		t.Login.Register(mux)
		t.Console.Register(mux)
		mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/console", http.StatusSeeOther) })
	}
	if t.Config.HasRole("scim") {
		t.SCIM.Register(mux)
	}
	for _, f := range t.mount {
		f(mux)
	}
	return httpx.WithRequestID(httpx.SecureHeaders(mux))
}

// accessOwner resolves an access token's user without requiring a usable project, for /oauth/userinfo.
func (t *Starlight) accessOwner(ctx context.Context, token string) (string, error) {
	var userID string
	err := t.Pool.QueryRow(ctx, `SELECT a.user_id FROM access_tokens a JOIN users u ON u.id = a.user_id
		WHERE a.token_hash = $1 AND a.revoked_at IS NULL AND a.expires_at > now() AND u.status = 'active'`, hashToken(token)).Scan(&userID)
	if err != nil {
		return "", authn.ErrInvalid
	}
	return userID, nil
}

func (t *Starlight) ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if n, err := db.Pending(ctx, t.Pool); err != nil || n > 0 {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}
	_, _ = w.Write([]byte("ready\n"))
}

// Close releases the pool.
func (t *Starlight) Close() { t.Pool.Close() }

func hostname() string {
	h, _ := os.Hostname()
	return strings.TrimSpace(h)
}

func version() string { return buildinfo.String() }
