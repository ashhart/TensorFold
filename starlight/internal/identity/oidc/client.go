package oidc

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	gooidc "github.com/coreos/go-oidc/v3/oidc"
	"golang.org/x/oauth2"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// authTimeSkew tolerates clock drift between Starlight and the IdP when enforcing max_age.
const authTimeSkew = 5 * time.Minute

// Errors that identify why a sign-in was refused.
var (
	ErrNoIDToken   = errors.New("oidc: token response carried no id_token")
	ErrNonce       = errors.New("oidc: nonce does not match this sign-in")
	ErrAuthorized  = errors.New("oidc: azp does not name this client")
	ErrAuthTime    = errors.New("oidc: authentication is older than max_age")
	ErrUserinfoSub = errors.New("oidc: userinfo subject differs from the ID token")
)

// Client is an OIDC client for one connection; cred is set when it authenticates with a certificate.
type Client struct {
	rp         *RP
	conn       connections.Connection
	cred       *Credential
	provider   *gooidc.Provider
	endSession string
	oauth      *oauth2.Config
}

// Client builds an OIDC client for c with the given redirect URL.
func (rp *RP) Client(ctx context.Context, c connections.Connection, secrets connections.Secrets, redirectURL string) (*Client, error) {
	p, endSession, err := rp.Provider(ctx, c)
	if err != nil {
		return nil, err
	}
	o := c.Config.OIDC
	ep := p.Endpoint()
	cfg := &oauth2.Config{ClientID: o.ClientID, RedirectURL: redirectURL, Scopes: o.Scopes, Endpoint: ep}
	var cred *Credential
	switch o.AuthMethod {
	case connections.AuthPost:
		cfg.ClientSecret, cfg.Endpoint.AuthStyle = secrets.ClientSecret, oauth2.AuthStyleInParams
	case connections.AuthPrivateJWT:
		cfg.Endpoint.AuthStyle = oauth2.AuthStyleInParams
		if cred, err = KeyCredential(o.ClientID, secrets.SigningKeyPEM, o.SigningCertificate, o.SigningKeyID, rp.now); err != nil {
			return nil, err
		}
	default:
		cfg.ClientSecret, cfg.Endpoint.AuthStyle = secrets.ClientSecret, oauth2.AuthStyleInHeader
	}
	return &Client{rp: rp, conn: c, cred: cred, provider: p, endSession: endSession, oauth: cfg}, nil
}

// AuthOptions are per-request authorization parameters.
type AuthOptions struct {
	LoginHint string
	Prompt    string
	MaxAge    int
}

// AuthURL returns the authorization endpoint URL with state, nonce and an S256 PKCE challenge.
func (c *Client) AuthURL(state, nonce, verifier string, o AuthOptions) string {
	opts := []oauth2.AuthCodeOption{gooidc.Nonce(nonce), oauth2.S256ChallengeOption(verifier)}
	for k, v := range c.conn.Config.OIDC.AuthParams {
		opts = append(opts, oauth2.SetAuthURLParam(k, v))
	}
	prompt := o.Prompt
	if prompt == "" {
		prompt = c.conn.Config.OIDC.Prompt
	}
	if prompt != "" {
		opts = append(opts, oauth2.SetAuthURLParam("prompt", prompt))
	}
	if o.LoginHint != "" {
		opts = append(opts, oauth2.SetAuthURLParam("login_hint", o.LoginHint))
	}
	if o.MaxAge > 0 {
		opts = append(opts, oauth2.SetAuthURLParam("max_age", strconv.Itoa(o.MaxAge)))
	}
	return c.oauth.AuthCodeURL(state, opts...)
}

// Result is a verified sign-in.
type Result struct {
	Subject   string
	Issuer    string
	Claims    map[string]any
	AuthTime  time.Time
	SessionID string
}

// Exchange redeems the code with the PKCE verifier and verifies the ID token against nonce and max_age.
func (c *Client) Exchange(ctx context.Context, code, pkceVerifier, nonce string, maxAge int) (*Result, error) {
	hctx := context.WithValue(ctx, oauth2.HTTPClient, c.rp.http)
	opts := []oauth2.AuthCodeOption{oauth2.VerifierOption(pkceVerifier)}
	if c.cred.Signed() {
		assertion, err := c.cred.Assertion(c.oauth.Endpoint.TokenURL)
		if err != nil {
			return nil, err
		}
		opts = append(opts, oauth2.SetAuthURLParam("client_assertion_type", JWTBearer), oauth2.SetAuthURLParam("client_assertion", assertion))
	}
	tok, err := c.oauth.Exchange(hctx, code, opts...)
	if err != nil {
		return nil, fmt.Errorf("oidc: code exchange: %w", err)
	}
	raw, _ := tok.Extra("id_token").(string)
	if raw == "" {
		return nil, ErrNoIDToken
	}
	idt, err := verifier(c.provider, c.oauth.ClientID, c.rp.now).Verify(ctx, raw)
	if err != nil {
		return nil, fmt.Errorf("oidc: id token: %w", err)
	}
	if subtle.ConstantTimeCompare([]byte(idt.Nonce), []byte(nonce)) != 1 || nonce == "" {
		return nil, ErrNonce
	}
	claims := map[string]any{}
	if err := idt.Claims(&claims); err != nil {
		return nil, err
	}
	if err := checkAZP(claims, idt.Audience, c.oauth.ClientID); err != nil {
		return nil, err
	}
	res := &Result{Subject: idt.Subject, Issuer: idt.Issuer, Claims: claims, SessionID: str(claims["sid"])}
	if at, ok := number(claims["auth_time"]); ok {
		res.AuthTime = time.Unix(int64(at), 0)
	}
	if maxAge > 0 {
		if res.AuthTime.IsZero() || c.rp.now().Sub(res.AuthTime) > time.Duration(maxAge)*time.Second+authTimeSkew {
			return nil, ErrAuthTime
		}
	}
	if c.conn.Config.OIDC.UseUserinfo && c.provider.UserInfoEndpoint() != "" {
		if err := c.mergeUserinfo(ctx, tok, res); err != nil {
			return nil, err
		}
	}
	return res, nil
}

func (c *Client) mergeUserinfo(ctx context.Context, tok *oauth2.Token, res *Result) error {
	ui, err := c.provider.UserInfo(c.rp.ctx(ctx), oauth2.StaticTokenSource(tok))
	if err != nil {
		return fmt.Errorf("oidc: userinfo: %w", err)
	}
	if subtle.ConstantTimeCompare([]byte(ui.Subject), []byte(res.Subject)) != 1 {
		return ErrUserinfoSub
	}
	extra := map[string]any{}
	if err := ui.Claims(&extra); err != nil {
		return err
	}
	for k, v := range extra {
		if _, taken := res.Claims[k]; !taken {
			res.Claims[k] = v
		}
	}
	return nil
}

// EndSessionURL is the IdP logout URL for RP-initiated logout, or empty when the IdP has none.
func (c *Client) EndSessionURL(postLogout string) string {
	if c.endSession == "" || !c.conn.Config.OIDC.RPLogout {
		return ""
	}
	return c.endSession + "?client_id=" + urlQuery(c.oauth.ClientID) + "&post_logout_redirect_uri=" + urlQuery(postLogout)
}

func checkAZP(claims map[string]any, aud []string, clientID string) error {
	azp, has := claims["azp"].(string)
	if has && azp != clientID {
		return ErrAuthorized
	}
	if len(aud) > 1 && azp != clientID {
		return ErrAuthorized
	}
	return nil
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func number(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	}
	return 0, false
}
