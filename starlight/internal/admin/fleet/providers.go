package fleet

import (
	"context"
	"encoding/json"
	"net/url"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// ProviderKind is a supported provider API with its documented base URL; an empty default must be entered.
type ProviderKind struct {
	Kind, Label, BaseURL string
}

// ProviderKinds lists supported provider kinds in display order.
var ProviderKinds = []ProviderKind{
	{"openai", "OpenAI", "https://api.openai.com/v1"},
	{"anthropic", "Anthropic", "https://api.anthropic.com"},
	{"azure_openai", "Azure OpenAI", ""},
	{"gemini", "Google Gemini (OpenAI-compatible)", "https://generativelanguage.googleapis.com/v1beta/openai"},
	{"openai_compatible", "OpenAI-compatible server", ""},
}

// DefaultBaseURL returns the documented base URL for a kind, or empty when the operator must supply one.
func DefaultBaseURL(kind string) string {
	for _, k := range ProviderKinds {
		if k.Kind == kind {
			return k.BaseURL
		}
	}
	return ""
}

// Provider is a frontier API account; its credential is write-only.
type Provider struct {
	ID            string
	Name          string
	Kind          string
	BaseURL       string
	HasCredential bool
	MaxClass      string
	Region        string
	Headers       map[string]string
	Internal      bool
	Enabled       bool
	Routes        int
}

// ProviderInput is what the provider form or configuration file sets; an empty Credential keeps the stored one.
type ProviderInput struct {
	Name            string
	Kind            string
	BaseURL         string
	Credential      string
	ClearCredential bool
	MaxClass        string
	Region          string
	Headers         map[string]string
	Internal        bool
	Enabled         bool
}

var reservedHeaders = map[string]bool{"authorization": true, "x-api-key": true, "api-key": true, "host": true, "content-length": true,
	"content-type": true, "cookie": true}

func (in *ProviderInput) validate() error {
	if in.BaseURL == "" {
		in.BaseURL = DefaultBaseURL(in.Kind)
	}
	if in.MaxClass == "" {
		in.MaxClass = "internal"
	}
	if err := slug("provider name", in.Name); err != nil {
		return err
	}
	if err := oneOf("kind", in.Kind, "openai", "anthropic", "azure_openai", "gemini", "openai_compatible"); err != nil {
		return err
	}
	u, err := url.Parse(in.BaseURL)
	if err != nil || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return Invalid("base URL must be an absolute URL without credentials, query or fragment")
	}
	if u.Scheme != "https" && !(u.Scheme == "http" && in.Internal) {
		return Invalid("base URL must use https; plain http is allowed only for providers marked internal")
	}
	if err := class(in.MaxClass); err != nil {
		return err
	}
	if len(in.Credential) > 8192 || strings.ContainsAny(in.Credential, "\r\n") {
		return Invalid("credential must be a single line")
	}
	if err := text("region", in.Region, 64, false); err != nil {
		return err
	}
	for k, v := range in.Headers {
		if reservedHeaders[strings.ToLower(k)] {
			return Invalid("header " + k + " is set by Starlight and cannot be overridden")
		}
		if k == "" || strings.ContainsAny(k, " :\r\n") || strings.ContainsAny(v, "\r\n") || len(k)+len(v) > 1024 {
			return Invalid("headers are Name: value pairs on one line each")
		}
	}
	return nil
}

func scanProvider(row interface{ Scan(...any) error }) (Provider, error) {
	var p Provider
	var headers []byte
	err := row.Scan(&p.ID, &p.Name, &p.Kind, &p.BaseURL, &p.HasCredential, &p.MaxClass, &p.Region, &headers, &p.Internal, &p.Enabled, &p.Routes)
	if err != nil {
		return p, notFound(err)
	}
	_ = json.Unmarshal(headers, &p.Headers)
	return p, nil
}

const providerCols = `p.id, p.name, p.kind, p.base_url, p.credential IS NOT NULL, p.max_classification, p.region, p.headers, p.internal, p.enabled,
	(SELECT count(*) FROM provider_routes r WHERE r.provider_id = p.id)`

// Providers lists provider accounts.
func Providers(ctx context.Context, q db.Querier) ([]Provider, error) {
	rows, err := q.Query(ctx, "SELECT "+providerCols+" FROM providers p ORDER BY p.name")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Provider
	for rows.Next() {
		p, err := scanProvider(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ProviderByName loads one provider.
func ProviderByName(ctx context.Context, q db.Querier, name string) (Provider, error) {
	return scanProvider(q.QueryRow(ctx, "SELECT "+providerCols+" FROM providers p WHERE p.name = $1", name))
}

// CreateProvider adds a provider and seals its credential.
func (t *Tx) CreateProvider(ctx context.Context, in ProviderInput) (Provider, error) {
	if err := in.validate(); err != nil {
		return Provider{}, err
	}
	id := ids.New()
	sealed, err := t.seal(id, in.Credential)
	if err != nil {
		return Provider{}, err
	}
	headers := marshalHeaders(in.Headers)
	_, err = t.tx.Exec(ctx, `INSERT INTO providers (id, name, kind, base_url, credential, max_classification, region, headers, internal, enabled)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`, id, in.Name, in.Kind, in.BaseURL, sealed, in.MaxClass, in.Region, headers, in.Internal, in.Enabled)
	if err != nil {
		return Provider{}, uniqueMsg(err, "a provider with that name already exists")
	}
	if err := t.record(ctx, "provider.created", "provider", id, map[string]any{"name": in.Name, "kind": in.Kind, "base_url": in.BaseURL,
		"max_classification": in.MaxClass, "internal": in.Internal, "credential_set": in.Credential != ""}); err != nil {
		return Provider{}, err
	}
	return ProviderByName(ctx, t.tx, in.Name)
}

// UpdateProvider changes a provider; the credential changes only when a new one is given or clearing is asked for.
func (t *Tx) UpdateProvider(ctx context.Context, id string, in ProviderInput) (Provider, error) {
	if err := in.validate(); err != nil {
		return Provider{}, err
	}
	headers := marshalHeaders(in.Headers)
	tag, err := t.tx.Exec(ctx, `UPDATE providers SET kind = $2, base_url = $3, max_classification = $4, region = $5, headers = $6, internal = $7,
		enabled = $8, updated_at = now() WHERE id = $1`, id, in.Kind, in.BaseURL, in.MaxClass, in.Region, headers, in.Internal, in.Enabled)
	if err != nil {
		return Provider{}, err
	}
	if tag.RowsAffected() == 0 {
		return Provider{}, ErrNotFound
	}
	credential := "kept"
	switch {
	case in.ClearCredential:
		credential = "cleared"
		_, err = t.tx.Exec(ctx, "UPDATE providers SET credential = NULL WHERE id = $1", id)
	case in.Credential != "":
		credential = "replaced"
		var sealed []byte
		if sealed, err = t.seal(id, in.Credential); err == nil {
			_, err = t.tx.Exec(ctx, "UPDATE providers SET credential = $2 WHERE id = $1", id, sealed)
		}
	}
	if err != nil {
		return Provider{}, err
	}
	if err := t.record(ctx, "provider.updated", "provider", id, map[string]any{"base_url": in.BaseURL, "max_classification": in.MaxClass,
		"internal": in.Internal, "enabled": in.Enabled, "credential": credential}); err != nil {
		return Provider{}, err
	}
	var name string
	if err := t.tx.QueryRow(ctx, "SELECT name FROM providers WHERE id = $1", id).Scan(&name); err != nil {
		return Provider{}, err
	}
	return ProviderByName(ctx, t.tx, name)
}

// CredentialMatches reports whether the stored credential equals plain, for idempotent configuration applies.
func (t *Tx) CredentialMatches(ctx context.Context, id, plain string) (bool, error) {
	var sealed []byte
	if err := t.tx.QueryRow(ctx, "SELECT credential FROM providers WHERE id = $1", id).Scan(&sealed); err != nil {
		return false, notFound(err)
	}
	if len(sealed) == 0 || t.keys == nil {
		return plain == "" && len(sealed) == 0, nil
	}
	got, err := t.keys.Open(sealed, "providers.credential:"+id)
	if err != nil {
		return false, err
	}
	return string(got) == plain, nil
}

func (t *Tx) seal(id, credential string) ([]byte, error) {
	if credential == "" {
		return nil, nil
	}
	if t.keys == nil {
		return nil, Invalid("credentials cannot be stored without the master key")
	}
	return t.keys.Seal([]byte(credential), "providers.credential:"+id)
}

func marshalHeaders(h map[string]string) []byte {
	if h == nil {
		h = map[string]string{}
	}
	b, _ := json.Marshal(h)
	return b
}

// ParseHeaders reads "Name: value" lines.
func ParseHeaders(lines []string) (map[string]string, error) {
	out := map[string]string{}
	for _, l := range lines {
		k, v, ok := strings.Cut(l, ":")
		if !ok {
			return nil, Invalid("headers are Name: value pairs on one line each")
		}
		out[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	return out, nil
}
