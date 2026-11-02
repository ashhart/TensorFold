package enrich

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

// GroupScope is the read-only Directory API scope the service account is delegated.
const GroupScope = "https://www.googleapis.com/auth/admin.directory.group.readonly"

// Google reads Workspace group membership through a service account with domain-wide delegation.
type Google struct {
	HTTP          *http.Client
	TokenURL      string
	DirectoryBase string
	now           func() time.Time
	mu            sync.Mutex
	tokens        map[string]cached
}

// NewGoogle returns a Directory API client for Google's public endpoints.
func NewGoogle(client *http.Client) *Google {
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	return &Google{HTTP: client, TokenURL: "https://oauth2.googleapis.com/token", DirectoryBase: "https://admin.googleapis.com",
		now: time.Now, tokens: map[string]cached{}}
}

type serviceAccount struct {
	Type         string `json:"type"`
	ClientEmail  string `json:"client_email"`
	PrivateKey   string `json:"private_key"`
	PrivateKeyID string `json:"private_key_id"`
}

// Groups lists the groups userEmail belongs to, following group-in-group membership up to depth levels.
func (g *Google) Groups(ctx context.Context, serviceAccountJSON, adminEmail, userEmail string, depth int) ([]directory.GroupRef, error) {
	tok, err := g.token(ctx, serviceAccountJSON, adminEmail)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var out []directory.GroupRef
	frontier := []string{strings.ToLower(userEmail)}
	calls := 0
	for level := 0; level <= depth && len(frontier) > 0; level++ {
		var next []string
		for _, key := range frontier {
			groups, n, err := g.memberOf(ctx, tok, key)
			calls += n
			if err != nil {
				return nil, err
			}
			if calls > 4*maxPages {
				return nil, errors.New("google: group expansion exceeds Starlight's limits")
			}
			for _, grp := range groups {
				if seen[grp.ExternalID] {
					continue
				}
				seen[grp.ExternalID] = true
				out = append(out, grp)
				next = append(next, grp.ExternalID)
			}
			if len(out) > maxGroups {
				return nil, errors.New("google: group membership exceeds Starlight's limits")
			}
		}
		frontier = next
	}
	return out, nil
}

func (g *Google) memberOf(ctx context.Context, token, userKey string) ([]directory.GroupRef, int, error) {
	var out []directory.GroupRef
	pageToken := ""
	for page := 1; ; page++ {
		if page > maxPages {
			return nil, page, errors.New("google: too many pages of groups")
		}
		q := url.Values{"userKey": {userKey}, "maxResults": {"200"}}
		if pageToken != "" {
			q.Set("pageToken", pageToken)
		}
		var body struct {
			Groups []struct {
				Email string `json:"email"`
				Name  string `json:"name"`
			} `json:"groups"`
			NextPageToken string `json:"nextPageToken"`
		}
		if err := getJSON(ctx, g.HTTP, g.DirectoryBase+"/admin/directory/v1/groups?"+q.Encode(), token, &body); err != nil {
			return nil, page, fmt.Errorf("google groups: %w", err)
		}
		for _, grp := range body.Groups {
			out = append(out, directory.GroupRef{ExternalID: strings.ToLower(grp.Email), DisplayName: grp.Name})
		}
		if body.NextPageToken == "" {
			return out, page, nil
		}
		pageToken = body.NextPageToken
	}
}

func (g *Google) token(ctx context.Context, keyJSON, adminEmail string) (string, error) {
	var sa serviceAccount
	if err := json.Unmarshal([]byte(keyJSON), &sa); err != nil || sa.Type != "service_account" || sa.ClientEmail == "" {
		return "", errors.New("google: the service account key is not a service_account JSON key")
	}
	if !strings.Contains(adminEmail, "@") {
		return "", errors.New("google: name the Workspace admin the service account acts as")
	}
	cacheKey := sa.ClientEmail + "|" + adminEmail
	g.mu.Lock()
	c, ok := g.tokens[cacheKey]
	g.mu.Unlock()
	if ok && time.Until(c.expires) > time.Minute {
		return c.token, nil
	}
	assertion, err := g.assertion(sa, adminEmail)
	if err != nil {
		return "", err
	}
	form := url.Values{"grant_type": {"urn:ietf:params:oauth:grant-type:jwt-bearer"}, "assertion": {assertion}}
	tok, ttl, err := postToken(ctx, g.HTTP, g.TokenURL, form)
	if err != nil {
		return "", fmt.Errorf("google token: %w", err)
	}
	g.mu.Lock()
	g.tokens[cacheKey] = cached{token: tok, expires: time.Now().Add(ttl)}
	g.mu.Unlock()
	return tok, nil
}

// assertion builds the RS256 JWT bearer grant that impersonates adminEmail with the group scope only.
func (g *Google) assertion(sa serviceAccount, adminEmail string) (string, error) {
	key, err := rsaKey(sa.PrivateKey)
	if err != nil {
		return "", err
	}
	now := g.now()
	header := map[string]string{"alg": "RS256", "typ": "JWT"}
	if sa.PrivateKeyID != "" {
		header["kid"] = sa.PrivateKeyID
	}
	claims := map[string]any{"iss": sa.ClientEmail, "sub": adminEmail, "scope": GroupScope, "aud": g.TokenURL,
		"iat": now.Unix(), "exp": now.Add(time.Hour).Unix()}
	h, _ := json.Marshal(header)
	c, _ := json.Marshal(claims)
	signing := base64.RawURLEncoding.EncodeToString(h) + "." + base64.RawURLEncoding.EncodeToString(c)
	sum := sha256.Sum256([]byte(signing))
	sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, sum[:])
	if err != nil {
		return "", err
	}
	return signing + "." + base64.RawURLEncoding.EncodeToString(sig), nil
}

func rsaKey(p string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(p))
	if block == nil {
		return nil, errors.New("google: service account private key is not PEM")
	}
	if k, err := x509.ParsePKCS8PrivateKey(block.Bytes); err == nil {
		if rk, ok := k.(*rsa.PrivateKey); ok {
			return rk, nil
		}
		return nil, errors.New("google: service account key is not RSA")
	}
	return x509.ParsePKCS1PrivateKey(block.Bytes)
}
