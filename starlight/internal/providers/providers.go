package providers

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
)

// Protocols a route can speak, keyed by the client-facing endpoint.
const (
	ChatCompletions = "openai.chat"
	Completions     = "openai.completions"
	Embeddings      = "openai.embeddings"
	Responses       = "openai.responses"
	Messages        = "anthropic.messages"
	CountTokens     = "anthropic.count_tokens"
)

// DefaultProtocols lists what each provider kind speaks natively.
var DefaultProtocols = map[string][]string{
	"openai":            {ChatCompletions, Completions, Embeddings, Responses},
	"azure_openai":      {ChatCompletions, Completions, Embeddings, Responses},
	"gemini":            {ChatCompletions, Embeddings},
	"anthropic":         {Messages, CountTokens, ChatCompletions},
	"openai_compatible": {ChatCompletions, Completions, Embeddings},
}

var paths = map[string]string{
	ChatCompletions: "/chat/completions",
	Completions:     "/completions",
	Embeddings:      "/embeddings",
	Responses:       "/responses",
	Messages:        "/v1/messages",
	CountTokens:     "/v1/messages/count_tokens",
}

// Credentials decrypts provider keys on demand and caches them briefly.
type Credentials struct {
	Pool *pgxpool.Pool
	Keys *keyring.Keyring
	mu   sync.Mutex
	hits map[string]cred
}

type cred struct {
	value   string
	headers map[string]string
	expires time.Time
}

func (c *Credentials) get(ctx context.Context, providerID string) (cred, error) {
	c.mu.Lock()
	if v, ok := c.hits[providerID]; ok && time.Now().Before(v.expires) {
		c.mu.Unlock()
		return v, nil
	}
	c.mu.Unlock()
	var sealed []byte
	var headers map[string]string
	if err := c.Pool.QueryRow(ctx, "SELECT credential, headers FROM providers WHERE id = $1", providerID).Scan(&sealed, &headers); err != nil {
		return cred{}, err
	}
	v := cred{headers: headers, expires: time.Now().Add(5 * time.Minute)}
	if len(sealed) > 0 {
		plain, err := c.Keys.Open(sealed, "providers.credential:"+providerID)
		if err != nil {
			return cred{}, err
		}
		v.value = string(plain)
	}
	c.mu.Lock()
	if c.hits == nil {
		c.hits = map[string]cred{}
	}
	c.hits[providerID] = v
	c.mu.Unlock()
	return v, nil
}

// Flush drops cached credentials after a provider changes.
func (c *Credentials) Flush() {
	c.mu.Lock()
	c.hits = nil
	c.mu.Unlock()
}

// Client sends requests to providers.
type Client struct {
	Egress *Egress
	Creds  *Credentials
}

// ErrProtocol means the route cannot serve the client's protocol.
var ErrProtocol = errors.New("providers: route does not speak this protocol")

// Do forwards body, already rewritten for the upstream model, to the route; the caller owns the response body.
func (c *Client) Do(ctx context.Context, route catalog.ProviderRoute, protocol string, body []byte, in http.Header) (*http.Response, error) {
	if !route.Speaks(protocol) {
		return nil, ErrProtocol
	}
	path, ok := paths[protocol]
	if !ok {
		return nil, ErrProtocol
	}
	if route.Kind == "anthropic" && protocol == ChatCompletions {
		path = "/v1/chat/completions"
	}
	cr, err := c.Creds.get(ctx, route.ProviderID)
	if err != nil {
		return nil, fmt.Errorf("providers: credential for %s: %w", route.ProviderName, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(route.BaseURL, "/")+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", in.Get("Accept"))
	for k, v := range cr.headers {
		if !reserved(k) {
			req.Header.Set(k, v)
		}
	}
	switch route.Kind {
	case "anthropic":
		req.Header.Set("x-api-key", cr.value)
		version := in.Get("anthropic-version")
		if version == "" {
			version = "2023-06-01"
		}
		req.Header.Set("anthropic-version", version)
		if beta := in.Get("anthropic-beta"); beta != "" {
			req.Header.Set("anthropic-beta", beta)
		}
	case "azure_openai":
		req.Header.Set("api-key", cr.value)
	default:
		if cr.value != "" {
			req.Header.Set("Authorization", "Bearer "+cr.value)
		}
	}
	return c.Egress.Client().Do(req)
}

func reserved(h string) bool {
	switch strings.ToLower(h) {
	case "authorization", "x-api-key", "api-key", "host", "content-length", "content-type", "cookie":
		return true
	}
	return false
}

// Fetch reads or deletes an object the provider stored, such as a Responses API response.
func (c *Client) Fetch(ctx context.Context, route catalog.ProviderRoute, method, path string) (*http.Response, error) {
	if !route.Speaks(Responses) {
		return nil, ErrProtocol
	}
	cr, err := c.Creds.get(ctx, route.ProviderID)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(route.BaseURL, "/")+path, nil)
	if err != nil {
		return nil, err
	}
	switch route.Kind {
	case "azure_openai":
		req.Header.Set("api-key", cr.value)
	default:
		req.Header.Set("Authorization", "Bearer "+cr.value)
	}
	return c.Egress.Client().Do(req)
}
