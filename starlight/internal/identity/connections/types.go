// Package connections stores the organisation's identity-provider connections and their sealed secrets.
package connections

import "time"

// Protocols a connection can speak.
const (
	OIDC = "oidc"
	SAML = "saml"
)

// Group strategies decide where a user's groups come from at sign-in.
const (
	GroupsNone            = "none"
	GroupsClaim           = "claim"
	GroupsEntraGraph      = "entra_graph"
	GroupsGoogleDirectory = "google_directory"
	GroupsSCIM            = "scim"
)

// OIDC client authentication methods at the token endpoint; private_key_jwt signs with a key and its certificate.
const (
	AuthBasic      = "client_secret_basic"
	AuthPost       = "client_secret_post"
	AuthPrivateJWT = "private_key_jwt"
)

// Connection is one identity provider the organisation signs in with.
type Connection struct {
	ID        string
	Slug      string
	Name      string
	Protocol  string
	Preset    string
	Domains   []string
	Enabled   bool
	JIT       bool
	Config    Config
	HasSecret bool
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Config is the non-secret part of a connection, stored as JSON.
type Config struct {
	OIDC     *OIDCConfig `json:"oidc,omitempty"`
	SAML     *SAMLConfig `json:"saml,omitempty"`
	Claims   ClaimMap    `json:"claims"`
	Groups   Groups      `json:"groups"`
	Policy   Policy      `json:"policy"`
	Provider Provider    `json:"provider"`
}

// OIDCConfig describes an OpenID Connect provider and Starlight's client registration there.
type OIDCConfig struct {
	Issuer             string            `json:"issuer"`
	ClientID           string            `json:"client_id"`
	AuthMethod         string            `json:"auth_method"`
	SigningKeyID       string            `json:"signing_key_id,omitempty"`
	SigningCertificate string            `json:"signing_certificate,omitempty"`
	Scopes             []string          `json:"scopes"`
	UseUserinfo        bool              `json:"use_userinfo,omitempty"`
	Prompt             string            `json:"prompt,omitempty"`
	AuthParams         map[string]string `json:"auth_params,omitempty"`
	RPLogout           bool              `json:"rp_logout,omitempty"`
	AcceptBearer       bool              `json:"accept_bearer,omitempty"`
	BearerAudiences    []string          `json:"bearer_audiences,omitempty"`
}

// SAMLConfig describes a SAML identity provider and Starlight's service-provider identity for it.
type SAMLConfig struct {
	MetadataURL       string `json:"metadata_url,omitempty"`
	MetadataXML       string `json:"metadata_xml,omitempty"`
	NameIDFormat      string `json:"name_id_format,omitempty"`
	AllowIDPInitiated bool   `json:"allow_idp_initiated,omitempty"`
	SignRequests      bool   `json:"sign_requests"`
	SPCertificate     string `json:"sp_certificate"`
	RPLogout          bool   `json:"rp_logout,omitempty"`
}

// ClaimMap names the claims or SAML attributes that carry each user property.
type ClaimMap struct {
	Subject       string `json:"subject,omitempty"`
	Email         string `json:"email,omitempty"`
	EmailVerified string `json:"email_verified,omitempty"`
	Name          string `json:"name,omitempty"`
	Username      string `json:"username,omitempty"`
	Groups        string `json:"groups,omitempty"`
	GroupsAreKeys bool   `json:"groups_are_keys,omitempty"`
	Tenant        string `json:"tenant,omitempty"`
}

// Groups configures where groups come from.
type Groups struct {
	Strategy    string `json:"strategy"`
	NestedDepth int    `json:"nested_depth,omitempty"`
}

// Policy restricts who may sign in through the connection.
type Policy struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	RequiredGroups []string `json:"required_groups,omitempty"`
	TrustEmail     bool     `json:"trust_email,omitempty"`
}

// Provider holds provider-specific settings that drive special checks.
type Provider struct {
	EntraTenantID    string   `json:"entra_tenant_id,omitempty"`
	GraphClientID    string   `json:"graph_client_id,omitempty"`
	GoogleDomains    []string `json:"google_domains,omitempty"`
	GoogleAdminEmail string   `json:"google_admin_email,omitempty"`
}

// Secrets are sealed at rest and only opened while talking to the provider.
type Secrets struct {
	ClientSecret         string `json:"client_secret,omitempty"`
	SigningKeyPEM        string `json:"signing_key_pem,omitempty"`
	SAMLKeyPEM           string `json:"saml_key_pem,omitempty"`
	GraphClientSecret    string `json:"graph_client_secret,omitempty"`
	GoogleServiceAccount string `json:"google_service_account,omitempty"`
}

// Merge copies every non-empty field of in over s, so write-only forms never erase stored secrets.
func (s Secrets) Merge(in Secrets) Secrets {
	pick := func(old, next string) string {
		if next != "" {
			return next
		}
		return old
	}
	return Secrets{
		ClientSecret:         pick(s.ClientSecret, in.ClientSecret),
		SigningKeyPEM:        pick(s.SigningKeyPEM, in.SigningKeyPEM),
		SAMLKeyPEM:           pick(s.SAMLKeyPEM, in.SAMLKeyPEM),
		GraphClientSecret:    pick(s.GraphClientSecret, in.GraphClientSecret),
		GoogleServiceAccount: pick(s.GoogleServiceAccount, in.GoogleServiceAccount),
	}
}

// Input creates or replaces a connection; empty secret fields keep the stored values.
type Input struct {
	Slug     string
	Name     string
	Protocol string
	Preset   string
	Domains  []string
	Enabled  bool
	JIT      bool
	Config   Config
	Secrets  Secrets
}

// Option is one sign-in button on the login page.
type Option struct {
	Slug     string
	Name     string
	Preset   string
	Protocol string
}
