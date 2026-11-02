// Package declare applies a fleet file to Starlight in one transaction, creating or updating only what the file names.
package declare

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"go.yaml.in/yaml/v3"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// File is a declarative fleet configuration.
type File struct {
	Version   int        `yaml:"version"`
	Settings  *Settings  `yaml:"settings"`
	Providers []Provider `yaml:"providers"`
	Models    []Model    `yaml:"models"`
	Projects  []Project  `yaml:"projects"`
	Limits    []Limit    `yaml:"limits"`
	OrgRoles  []Role     `yaml:"org_roles"`
}

// Settings are installation-wide settings; omitted ones stay as they are.
type Settings struct {
	OrgName            *string `yaml:"org_name"`
	Offline            *bool   `yaml:"offline"`
	AuditRetentionDays *int    `yaml:"audit_retention_days"`
}

// Provider is a frontier API account; Credential is a reference, env:NAME or file:/path, never the secret itself.
type Provider struct {
	Name       string            `yaml:"name"`
	Kind       string            `yaml:"kind"`
	BaseURL    string            `yaml:"base_url"`
	Credential string            `yaml:"credential"`
	MaxClass   string            `yaml:"max_classification"`
	Region     string            `yaml:"region"`
	Headers    map[string]string `yaml:"headers"`
	Internal   bool              `yaml:"internal"`
	Enabled    *bool             `yaml:"enabled"`
}

// Money is a dollar amount read from its literal text, so no float rounding touches prices.
type Money int64

// UnmarshalYAML parses amounts such as 0.15 or "$1.25" into micro-dollars.
func (m *Money) UnmarshalYAML(n *yaml.Node) error {
	v, err := fleet.ParseDollars("amount", n.Value)
	if err != nil {
		return fmt.Errorf("line %d: %w", n.Line, err)
	}
	*m = Money(v)
	return nil
}

// Prices are dollars per million tokens.
type Prices struct {
	Input  Money `yaml:"input"`
	Output Money `yaml:"output"`
	Cached Money `yaml:"cached"`
}

func (p Prices) fleet() fleet.Prices {
	return fleet.Prices{Input: int64(p.Input), Output: int64(p.Output), Cached: int64(p.Cached)}
}

// Capabilities describe what a model can do.
type Capabilities struct {
	ContextWindow int  `yaml:"context_window"`
	Tools         bool `yaml:"tools"`
	Vision        bool `yaml:"vision"`
	Reasoning     bool `yaml:"reasoning"`
}

// Model is a catalogue entry; Everyone grants or withdraws it for every project, within the caps beside it.
type Model struct {
	Name         string       `yaml:"name"`
	DisplayName  string       `yaml:"display_name"`
	Description  string       `yaml:"description"`
	Kind         string       `yaml:"kind"`
	RoutePolicy  string       `yaml:"route_policy"`
	Status       string       `yaml:"status"`
	Capabilities Capabilities `yaml:"capabilities"`
	Chargeback   Prices       `yaml:"chargeback"`
	Everyone     *bool        `yaml:"everyone"`
	MaxContext   *int         `yaml:"max_context"`
	MaxOutput    *int         `yaml:"max_output_tokens"`
	Aliases      []string     `yaml:"aliases"`
	Deployments  []Deployment `yaml:"deployments"`
	Routes       []Route      `yaml:"routes"`
}

// Grant gives a model to a project with optional caps.
type Grant struct {
	Model           string `yaml:"model"`
	MaxContext      *int   `yaml:"max_context"`
	MaxOutputTokens *int   `yaml:"max_output_tokens"`
}

// Spec is the typed engine spec, written with the same field names as the API.
type Spec struct{ wire.Spec }

// UnmarshalYAML decodes through the spec's JSON names and refuses unknown fields.
func (s *Spec) UnmarshalYAML(n *yaml.Node) error {
	var raw any
	if err := n.Decode(&raw); err != nil {
		return err
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return fmt.Errorf("line %d: spec: %w", n.Line, err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s.Spec); err != nil {
		return fmt.Errorf("line %d: spec: %w", n.Line, err)
	}
	return nil
}

// Deployment runs a model on nodes; MinReplicas defaults to 1 and Status, when set, is active, paused or retired.
type Deployment struct {
	Name        string            `yaml:"name"`
	Spec        Spec              `yaml:"spec"`
	Pool        string            `yaml:"pool"`
	Selector    map[string]string `yaml:"selector"`
	Mode        string            `yaml:"mode"`
	MinReplicas *int              `yaml:"min_replicas"`
	MaxReplicas int               `yaml:"max_replicas"`
	IdleSeconds int               `yaml:"idle_seconds"`
	Status      string            `yaml:"status"`
}

// Route sends a model to a provider's upstream model.
type Route struct {
	Provider       string   `yaml:"provider"`
	UpstreamModel  string   `yaml:"upstream_model"`
	Protocols      []string `yaml:"protocols"`
	Priority       int      `yaml:"priority"`
	Prices         Prices   `yaml:"prices"`
	MaxConcurrency int      `yaml:"max_concurrency"`
	Enabled        *bool    `yaml:"enabled"`
}

// Project is a unit of access and budget with its members, models, aliases, limits and service accounts.
type Project struct {
	Slug            string            `yaml:"slug"`
	Name            string            `yaml:"name"`
	Description     string            `yaml:"description"`
	Classification  string            `yaml:"classification"`
	MaxPriority     string            `yaml:"max_priority"`
	PlanApproval    string            `yaml:"plan_approval"`
	SecretsGuard    string            `yaml:"secrets_guard"`
	IsolateCache    bool              `yaml:"isolate_prompt_cache"`
	Everyone        bool              `yaml:"everyone"`
	Archived        bool              `yaml:"archived"`
	Members         []Role            `yaml:"members"`
	Models          []Grant           `yaml:"models"`
	Aliases         map[string]string `yaml:"aliases"`
	Limits          []Limit           `yaml:"limits"`
	ServiceAccounts []ServiceAccount  `yaml:"service_accounts"`
}

// ServiceAccount is a non-human identity in a project; its keys come from the console or CLI, never from this file.
type ServiceAccount struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// Group names an identity provider group by connection slug and the group's id at the provider.
type Group struct {
	Connection string `yaml:"connection"`
	ID         string `yaml:"id"`
}

// Role binds a role to one subject: a user by email, an identity provider group, or a project's service account.
type Role struct {
	Role           string `yaml:"role"`
	User           string `yaml:"user"`
	Group          *Group `yaml:"group"`
	ServiceAccount string `yaml:"service_account"`
}

// Limit caps usage; Scope is project or member inside a project, and organisation-wide at the top level.
type Limit struct {
	Kind       string `yaml:"kind"`
	Value      string `yaml:"value"`
	Model      string `yaml:"model"`
	Scope      string `yaml:"scope"`
	BoostValue string `yaml:"boost_value"`
	BoostUntil string `yaml:"boost_until"`
}

// Load reads a fleet file strictly: unknown keys, a second document and a missing version are errors.
func Load(r io.Reader) (*File, error) {
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	var f File
	if err := dec.Decode(&f); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("the fleet file is empty")
		}
		return nil, hint(err)
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, errors.New("the fleet file must hold exactly one YAML document")
	}
	if f.Version != 1 {
		return nil, errors.New("the fleet file must start with version: 1")
	}
	return &f, nil
}

// LoadFile reads a fleet file from disk.
func LoadFile(path string) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	return Load(fh)
}

// hint explains the common mistake of passing the server configuration instead of a fleet file.
func hint(err error) error {
	for _, k := range []string{"database", "listen", "public_url", "node_url", "keys", "tls", "identity", "gateway"} {
		if strings.Contains(err.Error(), "field "+k+" not found") {
			return fmt.Errorf("%w (this looks like the server configuration; starlight apply takes a fleet file such as deploy/examples/fleet.yaml)", err)
		}
	}
	return err
}
