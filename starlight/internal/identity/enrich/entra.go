// Package enrich reads group membership from directory APIs when an IdP does not put it in tokens.
package enrich

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
)

// Limits that keep a hostile or huge directory from exhausting a sign-in.
const (
	maxPages  = 50
	maxGroups = 5000
	maxBody   = 4 << 20
)

var guid = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// Entra reads a user's transitive group membership from Microsoft Graph with a client secret or certificate.
type Entra struct {
	HTTP      *http.Client
	LoginBase string
	GraphBase string
	mu        sync.Mutex
	tokens    map[string]cached
}

type cached struct {
	token   string
	expires time.Time
}

// NewEntra returns a Graph client for the public Microsoft cloud.
func NewEntra(client *http.Client) *Entra {
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	return &Entra{HTTP: client, LoginBase: "https://login.microsoftonline.com", GraphBase: "https://graph.microsoft.com", tokens: map[string]cached{}}
}

// Groups lists the security and Microsoft 365 groups the user belongs to, nested ones included.
func (e *Entra) Groups(ctx context.Context, tenantID string, cred *oidc.Credential, userOID string) ([]directory.GroupRef, error) {
	tenantID, userOID = strings.ToLower(tenantID), strings.ToLower(userOID)
	if !guid.MatchString(tenantID) || !guid.MatchString(userOID) {
		return nil, errors.New("entra: tenant and user ids must be GUIDs")
	}
	if !cred.Configured() {
		return nil, errors.New("entra: Graph credentials are not configured")
	}
	tok, err := e.token(ctx, tenantID, cred)
	if err != nil {
		return nil, err
	}
	next := e.GraphBase + "/v1.0/users/" + userOID + "/transitiveMemberOf/microsoft.graph.group?$select=id,displayName&$top=999"
	var out []directory.GroupRef
	for page := 0; next != ""; page++ {
		if page >= maxPages || len(out) >= maxGroups {
			return nil, errors.New("entra: group membership exceeds Starlight's limits")
		}
		if err := sameOrigin(next, e.GraphBase); err != nil {
			return nil, err
		}
		var body struct {
			Value []struct {
				ID          string `json:"id"`
				DisplayName string `json:"displayName"`
			} `json:"value"`
			Next string `json:"@odata.nextLink"`
		}
		if err := getJSON(ctx, e.HTTP, next, tok, &body); err != nil {
			return nil, fmt.Errorf("entra groups: %w", err)
		}
		for _, g := range body.Value {
			out = append(out, directory.GroupRef{ExternalID: strings.ToLower(g.ID), DisplayName: g.DisplayName})
		}
		next = body.Next
	}
	return out, nil
}

func (e *Entra) token(ctx context.Context, tenantID string, cred *oidc.Credential) (string, error) {
	key := tenantID + "/" + cred.ClientID
	e.mu.Lock()
	c, ok := e.tokens[key]
	e.mu.Unlock()
	if ok && time.Until(c.expires) > time.Minute {
		return c.token, nil
	}
	tokenURL := e.LoginBase + "/" + tenantID + "/oauth2/v2.0/token"
	form := url.Values{"grant_type": {"client_credentials"}, "scope": {strings.TrimRight(e.GraphBase, "/") + "/.default"}}
	if err := cred.Apply(form, tokenURL); err != nil {
		return "", fmt.Errorf("entra token: %w", err)
	}
	tok, ttl, err := postToken(ctx, e.HTTP, tokenURL, form)
	if err != nil {
		return "", fmt.Errorf("entra token: %w", err)
	}
	e.mu.Lock()
	e.tokens[key] = cached{token: tok, expires: time.Now().Add(ttl)}
	e.mu.Unlock()
	return tok, nil
}

func postToken(ctx context.Context, client *http.Client, tokenURL string, form url.Values) (string, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	var body struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int64  `json:"expires_in"`
		Error       string `json:"error"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxBody)).Decode(&body); err != nil {
		return "", 0, fmt.Errorf("token endpoint returned %s", resp.Status)
	}
	if resp.StatusCode != http.StatusOK || body.AccessToken == "" {
		return "", 0, fmt.Errorf("token endpoint returned %s %s", resp.Status, body.Error)
	}
	ttl := time.Duration(body.ExpiresIn) * time.Second
	if ttl <= 0 || ttl > 2*time.Hour {
		ttl = 30 * time.Minute
	}
	return body.AccessToken, ttl, nil
}

func getJSON(ctx context.Context, client *http.Client, target, token string, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s returned %s", req.URL.Path, resp.Status)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, maxBody)).Decode(v)
}

// sameOrigin stops a paging link from sending the access token to another host.
func sameOrigin(link, base string) error {
	a, err := url.Parse(link)
	if err != nil {
		return err
	}
	b, err := url.Parse(base)
	if err != nil {
		return err
	}
	if a.Scheme != b.Scheme || a.Host != b.Host {
		return errors.New("enrich: paging link leaves the directory API host")
	}
	return nil
}
