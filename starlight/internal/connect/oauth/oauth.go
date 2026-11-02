// Package oauth speaks Starlight's OAuth 2.0 token, refresh and revocation endpoints as a public client.
package oauth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

// Scope is what Connect asks Starlight for.
const Scope = discovery.ScopeInference + " " + discovery.ScopeOffline + " " + discovery.ScopeProfile

// Error is an OAuth error reply such as invalid_grant or authorization_pending.
type Error struct {
	Code        string `json:"error"`
	Description string `json:"error_description"`
	Status      int    `json:"-"`
}

func (e *Error) Error() string {
	if e.Description != "" {
		return e.Code + ": " + e.Description
	}
	return e.Code
}

// Is reports whether err is an OAuth error with the given code.
func Is(err error, code string) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == code
}

// Post sends a form to an OAuth endpoint and decodes a token reply or an OAuth error.
func Post(ctx context.Context, client *http.Client, endpoint string, form url.Values, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		e := &Error{Status: resp.StatusCode}
		if json.Unmarshal(body, e) != nil || e.Code == "" {
			e.Code, e.Description = "server_error", resp.Status
		}
		return e
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("token endpoint reply: %w", err)
	}
	return nil
}

// Token posts a grant and checks that the reply carries a bearer access token.
func Token(ctx context.Context, client *http.Client, endpoint string, form url.Values) (discovery.TokenResponse, error) {
	var tr discovery.TokenResponse
	if err := Post(ctx, client, endpoint, form, &tr); err != nil {
		return tr, err
	}
	if tr.AccessToken == "" || !strings.EqualFold(tr.TokenType, "bearer") {
		return tr, errors.New("token endpoint returned no bearer access token")
	}
	return tr, nil
}

// Refresh exchanges a refresh token; Starlight rotates it, so the reply's refresh token replaces the old one.
func Refresh(ctx context.Context, client *http.Client, doc discovery.Document, refreshToken string) (discovery.TokenResponse, error) {
	return Token(ctx, client, doc.TokenEndpoint, url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {discovery.ClientConnect},
	})
}

// Revoke asks Starlight to revoke a token; RFC 7009 servers answer 200 even for unknown tokens.
func Revoke(ctx context.Context, client *http.Client, doc discovery.Document, token, hint string) error {
	return Post(ctx, client, doc.RevocationEndpoint, url.Values{
		"token":           {token},
		"token_type_hint": {hint},
		"client_id":       {discovery.ClientConnect},
	}, nil)
}

// PKCE returns a fresh code verifier and its S256 challenge.
func PKCE() (verifier, challenge string) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		panic("oauth: crypto/rand failed: " + err.Error())
	}
	verifier = base64.RawURLEncoding.EncodeToString(b)
	sum := sha256.Sum256([]byte(verifier))
	return verifier, base64.RawURLEncoding.EncodeToString(sum[:])
}

// State returns an unguessable value binding a browser redirect to the flow that started it.
func State() string {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		panic("oauth: crypto/rand failed: " + err.Error())
	}
	return base64.RawURLEncoding.EncodeToString(b)
}
