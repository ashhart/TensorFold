// Package discovery is the public document clients read to find Starlight's endpoints.
package discovery

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Path is where Starlight serves its discovery document.
const Path = "/.well-known/tensorfold-starlight"

// Built-in OAuth clients; both are public clients that must use PKCE or the device grant.
const (
	ClientConnect = "tensorfold-connect"
	ClientCLI     = "tensorfold-cli"
)

// Scopes a Starlight access token can carry.
const (
	ScopeInference = "inference"
	ScopeOffline   = "offline_access"
	ScopeProfile   = "profile"
)

// Document lists Starlight's endpoints; it contains nothing secret.
type Document struct {
	Issuer                      string `json:"issuer"`
	OrgName                     string `json:"org_name"`
	Version                     string `json:"version"`
	InstallationID              string `json:"installation_id"`
	Offline                     bool   `json:"offline"`
	AuthorizationEndpoint       string `json:"authorization_endpoint"`
	TokenEndpoint               string `json:"token_endpoint"`
	DeviceAuthorizationEndpoint string `json:"device_authorization_endpoint"`
	RevocationEndpoint          string `json:"revocation_endpoint"`
	UserinfoEndpoint            string `json:"userinfo_endpoint"`
	APIBase                     string `json:"api_base"`
	ConsoleURL                  string `json:"console_url"`
}

// Build fills endpoint URLs from the public base URL.
func Build(base, org, version, installation string, offline bool) Document {
	base = strings.TrimRight(base, "/")
	return Document{
		Issuer: base, OrgName: org, Version: version, InstallationID: installation, Offline: offline,
		AuthorizationEndpoint:       base + "/oauth/authorize",
		TokenEndpoint:               base + "/oauth/token",
		DeviceAuthorizationEndpoint: base + "/oauth/device/code",
		RevocationEndpoint:          base + "/oauth/revoke",
		UserinfoEndpoint:            base + "/oauth/userinfo",
		APIBase:                     base + "/v1",
		ConsoleURL:                  base + "/console",
	}
}

// Fetch reads and checks the document; every endpoint must share the issuer's origin.
func Fetch(ctx context.Context, client *http.Client, base string) (Document, error) {
	var d Document
	u, err := url.Parse(strings.TrimRight(base, "/"))
	if err != nil || u.Scheme == "" || u.Host == "" {
		return d, fmt.Errorf("server url %q is not absolute", base)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String()+Path, nil)
	if err != nil {
		return d, err
	}
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		return d, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return d, fmt.Errorf("discovery returned %s", resp.Status)
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&d); err != nil {
		return d, fmt.Errorf("discovery document: %w", err)
	}
	if strings.TrimRight(d.Issuer, "/") != u.String() {
		return d, fmt.Errorf("discovery issuer %q does not match %q", d.Issuer, u.String())
	}
	for _, e := range []string{d.AuthorizationEndpoint, d.TokenEndpoint, d.DeviceAuthorizationEndpoint, d.RevocationEndpoint, d.UserinfoEndpoint, d.APIBase} {
		eu, err := url.Parse(e)
		if err != nil || eu.Scheme != u.Scheme || eu.Host != u.Host {
			return d, errors.New("discovery endpoint outside the issuer origin")
		}
	}
	return d, nil
}

// TokenResponse is the OAuth 2.0 token endpoint reply.
type TokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int    `json:"expires_in"`
	RefreshToken string `json:"refresh_token,omitempty"`
	Scope        string `json:"scope"`
}

// DeviceResponse is the RFC 8628 device authorization reply.
type DeviceResponse struct {
	DeviceCode              string `json:"device_code"`
	UserCode                string `json:"user_code"`
	VerificationURI         string `json:"verification_uri"`
	VerificationURIComplete string `json:"verification_uri_complete"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
}

// Me is the signed-in user's view: who they are, their projects and usable models.
type Me struct {
	UserID      string    `json:"user_id"`
	Email       string    `json:"email"`
	DisplayName string    `json:"display_name"`
	OrgName     string    `json:"org_name"`
	Projects    []Project `json:"projects"`
}

// Project is one project the user can act in.
type Project struct {
	ID             string   `json:"id"`
	Slug           string   `json:"slug"`
	Name           string   `json:"name"`
	Role           string   `json:"role"`
	Classification string   `json:"classification"`
	Default        bool     `json:"default"`
	Models         []string `json:"models"`
}
