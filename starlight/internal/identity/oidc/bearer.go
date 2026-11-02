package oidc

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	gooidc "github.com/coreos/go-oidc/v3/oidc"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// Errors returned by bearer and logout verification.
var (
	ErrNoBearerIssuer = errors.New("oidc: no connection accepts bearer tokens from this issuer")
	ErrAudience       = errors.New("oidc: bearer token audience is not accepted")
	ErrNotAccessToken = errors.New("oidc: bearer token is not an access token")
	ErrLogoutStale    = errors.New("oidc: logout token issued outside the accepted window")
	ErrMalformedJWT   = errors.New("oidc: malformed JWT")
)

// maxJWT bounds the size of a token Starlight will parse.
const maxJWT = 16 << 10

// VerifyLogout validates an OIDC back-channel logout token; the caller still enforces jti replay.
func (rp *RP) VerifyLogout(ctx context.Context, c connections.Connection, raw string) (*gooidc.LogoutToken, error) {
	if len(raw) > maxJWT {
		return nil, ErrMalformedJWT
	}
	p, _, err := rp.Provider(ctx, c)
	if err != nil {
		return nil, err
	}
	lt, err := verifier(p, c.Config.OIDC.ClientID, rp.now).VerifyLogout(ctx, raw)
	if err != nil {
		return nil, err
	}
	now := rp.now()
	if lt.IssuedAt.IsZero() || lt.IssuedAt.After(now.Add(2*time.Minute)) || now.Sub(lt.IssuedAt) > 10*time.Minute {
		return nil, ErrLogoutStale
	}
	return lt, nil
}

// Bearer verifies JWT access tokens issued by connections that opted in to bearer authentication.
type Bearer struct {
	rp     *RP
	store  *connections.Store
	mu     sync.Mutex
	loaded time.Time
	conns  []connections.Connection
}

// NewBearer returns a verifier over the store's opted-in connections.
func NewBearer(rp *RP, store *connections.Store) *Bearer {
	return &Bearer{rp: rp, store: store}
}

func (b *Bearer) candidates(ctx context.Context) ([]connections.Connection, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.rp.now().Sub(b.loaded) < 30*time.Second {
		return b.conns, nil
	}
	all, err := b.store.List(ctx)
	if err != nil {
		return nil, err
	}
	conns := make([]connections.Connection, 0, len(all))
	for _, c := range all {
		if c.Enabled && c.Protocol == connections.OIDC && c.Config.OIDC != nil && c.Config.OIDC.AcceptBearer {
			conns = append(conns, c)
		}
	}
	b.conns, b.loaded = conns, b.rp.now()
	return b.conns, nil
}

// Verify checks signature, issuer, expiry and audience and returns the connection that issued the token.
func (b *Bearer) Verify(ctx context.Context, raw string) (connections.Connection, *Result, error) {
	iss, err := PeekIssuer(raw)
	if err != nil {
		return connections.Connection{}, nil, err
	}
	conns, err := b.candidates(ctx)
	if err != nil {
		return connections.Connection{}, nil, err
	}
	for _, c := range conns {
		if strings.TrimRight(c.Config.OIDC.Issuer, "/") != strings.TrimRight(iss, "/") {
			continue
		}
		p, _, err := b.rp.Provider(ctx, c)
		if err != nil {
			return c, nil, err
		}
		v := p.Verifier(&gooidc.Config{SkipClientIDCheck: true, SupportedSigningAlgs: SigningAlgs, Now: b.rp.now})
		t, err := v.Verify(ctx, raw)
		if err != nil {
			return c, nil, fmt.Errorf("oidc: bearer: %w", err)
		}
		if !intersects(t.Audience, c.Config.OIDC.BearerAudiences) {
			return c, nil, ErrAudience
		}
		claims := map[string]any{}
		if err := t.Claims(&claims); err != nil {
			return c, nil, err
		}
		if claims["scp"] == nil && claims["scope"] == nil && claims["roles"] == nil {
			return c, nil, ErrNotAccessToken
		}
		return c, &Result{Subject: t.Subject, Issuer: t.Issuer, Claims: claims}, nil
	}
	return connections.Connection{}, nil, ErrNoBearerIssuer
}

// PeekIssuer reads iss from an unverified JWT so the right connection can verify it.
func PeekIssuer(raw string) (string, error) {
	if len(raw) > maxJWT {
		return "", ErrMalformedJWT
	}
	parts := strings.Split(raw, ".")
	if len(parts) != 3 {
		return "", ErrMalformedJWT
	}
	body, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", ErrMalformedJWT
	}
	var c struct {
		Iss string `json:"iss"`
	}
	if err := json.Unmarshal(body, &c); err != nil || c.Iss == "" {
		return "", ErrMalformedJWT
	}
	return c.Iss, nil
}

// LooksLikeJWT reports whether a bearer credential has the three-part JWT shape.
func LooksLikeJWT(raw string) bool {
	return strings.Count(raw, ".") == 2 && strings.HasPrefix(raw, "eyJ")
}

func intersects(have, want []string) bool {
	for _, a := range have {
		for _, b := range want {
			if a == b {
				return true
			}
		}
	}
	return false
}
