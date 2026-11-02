// Package oidc is Starlight's OpenID Connect relying party: discovery, code flow, token checks and logout tokens.
package oidc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	gooidc "github.com/coreos/go-oidc/v3/oidc"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// SigningAlgs are the only JWS algorithms Starlight accepts; none and HMAC are never allowed.
var SigningAlgs = []string{
	gooidc.RS256, gooidc.RS384, gooidc.RS512, gooidc.ES256, gooidc.ES384, gooidc.ES512,
	gooidc.PS256, gooidc.PS384, gooidc.PS512, gooidc.EdDSA,
}

// providerTTL bounds how long discovery metadata is reused before it is fetched again.
const providerTTL = 24 * time.Hour

// RP caches provider discovery per connection and builds clients from it.
type RP struct {
	http  *http.Client
	now   func() time.Time
	mu    sync.Mutex
	cache map[string]*entry
}

type entry struct {
	issuer     string
	updated    time.Time
	loaded     time.Time
	provider   *gooidc.Provider
	endSession string
}

// NewRP returns a relying party that reaches providers with client.
func NewRP(client *http.Client, now func() time.Time) *RP {
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	if now == nil {
		now = time.Now
	}
	return &RP{http: client, now: now, cache: map[string]*entry{}}
}

func (rp *RP) ctx(ctx context.Context) context.Context {
	return gooidc.ClientContext(ctx, rp.http)
}

// Provider returns discovery for the connection, tolerating an issuer that differs only by a trailing slash.
func (rp *RP) Provider(ctx context.Context, c connections.Connection) (*gooidc.Provider, string, error) {
	if c.Config.OIDC == nil {
		return nil, "", errors.New("oidc: connection has no OIDC settings")
	}
	issuer := c.Config.OIDC.Issuer
	rp.mu.Lock()
	e := rp.cache[c.ID]
	rp.mu.Unlock()
	if e != nil && e.issuer == issuer && e.updated.Equal(c.UpdatedAt) && rp.now().Sub(e.loaded) < providerTTL {
		return e.provider, e.endSession, nil
	}
	p, err := gooidc.NewProvider(rp.ctx(ctx), issuer)
	var mismatch *gooidc.IssuerMismatchError
	if errors.As(err, &mismatch) && strings.TrimRight(mismatch.Discovered, "/") == strings.TrimRight(issuer, "/") {
		p, err = gooidc.NewProvider(rp.ctx(ctx), mismatch.Discovered)
	}
	if err != nil {
		return nil, "", fmt.Errorf("oidc discovery for %s: %w", issuer, err)
	}
	if err := checkEndpoints(p); err != nil {
		return nil, "", err
	}
	var meta struct {
		EndSession string `json:"end_session_endpoint"`
	}
	_ = p.Claims(&meta)
	rp.mu.Lock()
	rp.cache[c.ID] = &entry{issuer: issuer, updated: c.UpdatedAt, loaded: rp.now(), provider: p, endSession: meta.EndSession}
	rp.mu.Unlock()
	return p, meta.EndSession, nil
}

// Forget drops cached discovery, for example after a connection is edited or deleted.
func (rp *RP) Forget(connectionID string) {
	rp.mu.Lock()
	delete(rp.cache, connectionID)
	rp.mu.Unlock()
}

func checkEndpoints(p *gooidc.Provider) error {
	for name, raw := range map[string]string{"authorization_endpoint": p.Endpoint().AuthURL, "token_endpoint": p.Endpoint().TokenURL} {
		u, err := url.Parse(raw)
		if err != nil || u.Host == "" {
			return fmt.Errorf("oidc: discovery %s is not a URL", name)
		}
		if u.Scheme != "https" && !(u.Scheme == "http" && loopback(u.Hostname())) {
			return fmt.Errorf("oidc: discovery %s must use https", name)
		}
	}
	return nil
}

func loopback(host string) bool {
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// verifier builds an ID token verifier pinned to the client and the allowed algorithms.
func verifier(p *gooidc.Provider, clientID string, now func() time.Time) *gooidc.IDTokenVerifier {
	return p.Verifier(&gooidc.Config{ClientID: clientID, SupportedSigningAlgs: SigningAlgs, Now: now})
}
