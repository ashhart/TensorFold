// Package presets is the catalogue of identity providers Starlight knows how to configure.
package presets

import (
	"sort"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// Field kinds tell forms how to render and parse a value.
const (
	KindText     = "text"
	KindURL      = "url"
	KindSecret   = "secret"
	KindTextarea = "textarea"
	KindList     = "list"
	KindBool     = "bool"
	KindSelect   = "select"
)

// Field is one input a preset needs from the administrator; WriteOnly values are never shown again.
type Field struct {
	Name        string
	Label       string
	Help        string
	Kind        string
	Protocol    string
	Required    bool
	Placeholder string
	Options     []string
	WriteOnly   bool
	ConsoleOnly bool
}

// Preset describes how to talk to one identity provider product.
type Preset struct {
	Name      string
	Label     string
	Summary   string
	Protocols []string
	Doc       string
	Fields    []Field
	issuer    func(Values) (string, error)
	oidc      func(*connections.Input, Values)
	saml      func(*connections.Input, Values)
}

// Supports reports whether the preset can use protocol.
func (p Preset) Supports(protocol string) bool {
	for _, x := range p.Protocols {
		if x == protocol {
			return true
		}
	}
	return false
}

// FieldsFor returns the preset's fields that apply to protocol, including the common ones.
func (p Preset) FieldsFor(protocol string) []Field {
	var out []Field
	for _, f := range append(common(protocol), p.Fields...) {
		if f.Protocol == "" || f.Protocol == protocol {
			out = append(out, f)
		}
	}
	return out
}

var catalog = map[string]Preset{}

func register(p Preset) {
	if p.Doc == "" {
		p.Doc = "docs/sso/" + p.Name + ".md"
	}
	catalog[p.Name] = p
}

// Get returns the preset with this name.
func Get(name string) (Preset, bool) {
	p, ok := catalog[name]
	return p, ok
}

// All returns every preset, generic ones last.
func All() []Preset {
	out := make([]Preset, 0, len(catalog))
	for _, p := range catalog {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		gi, gj := isGeneric(out[i].Name), isGeneric(out[j].Name)
		if gi != gj {
			return gj
		}
		return out[i].Label < out[j].Label
	})
	return out
}

func isGeneric(name string) bool { return name == "generic_oidc" || name == "generic_saml" }

func common(protocol string) []Field {
	fs := []Field{
		{Name: "name", Label: "Display name", Kind: KindText, Required: true, Placeholder: "Example Corp SSO"},
		{Name: "slug", Label: "URL slug", Kind: KindText, Required: true, Placeholder: "corp", Help: "Lowercase; appears in redirect URLs."},
		{Name: "domains", Label: "Email domains", Kind: KindList, Help: "Users with these email domains are sent to this connection."},
		{Name: "jit", Label: "Create users on first sign-in", Kind: KindBool, Help: "Off when SCIM provisions every user."},
		{Name: "required_groups", Label: "Required groups", Kind: KindList, Help: "Only members of at least one of these groups may sign in."},
	}
	if protocol == connections.OIDC {
		return append(fs,
			Field{Name: "client_id", Label: "Client ID", Kind: KindText, Required: true, Protocol: connections.OIDC},
			Field{Name: "client_secret", Label: "Client secret", Kind: KindSecret, Protocol: connections.OIDC},
			Field{Name: "rp_logout", Label: "Sign out of the IdP too", Kind: KindBool, Protocol: connections.OIDC},
			Field{Name: "accept_bearer", Label: "Accept IdP access tokens on the API", Kind: KindBool, Protocol: connections.OIDC},
			Field{Name: "bearer_audiences", Label: "Access token audiences", Kind: KindList, Protocol: connections.OIDC,
				Help: "The aud your IdP puts in access tokens for Starlight."},
		)
	}
	return append(fs,
		Field{Name: "metadata_url", Label: "IdP metadata URL", Kind: KindURL, Protocol: connections.SAML},
		Field{Name: "metadata_xml", Label: "IdP metadata XML", Kind: KindTextarea, Protocol: connections.SAML, Help: "Paste when the IdP gives a file."},
		Field{Name: "allow_idp_initiated", Label: "Allow IdP-initiated sign-in", Kind: KindBool, Protocol: connections.SAML,
			Help: "Off unless your portal starts sign-ins; each assertion still works once."},
		Field{Name: "rp_logout", Label: "Sign out of the IdP too", Kind: KindBool, Protocol: connections.SAML,
			Help: "Sends a signed logout request when the IdP metadata lists a single logout endpoint."},
	)
}

func join(parts ...[]Field) []Field {
	var out []Field
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// clientAuth picks how Starlight authenticates at the token endpoint: a client secret or certificate (private_key_jwt).
func clientAuth(keyID bool) []Field {
	fs := []Field{
		{Name: "auth_method", Label: "Client authentication", Kind: KindSelect, Protocol: connections.OIDC,
			Options: []string{connections.AuthBasic, connections.AuthPost, connections.AuthPrivateJWT},
			Help:    "private_key_jwt signs every token request with a certificate's key instead of sending the client secret."},
		{Name: "generate_certificate", Label: "Generate a key and certificate", Kind: KindBool, Protocol: connections.OIDC, ConsoleOnly: true,
			Help: "Starlight seals the key and offers the certificate for download, to upload at the provider."},
		{Name: "signing_certificate", Label: "Certificate (PEM)", Kind: KindTextarea, Protocol: connections.OIDC,
			Help: "Or paste your own certificate here, and its private key in the next field."},
		{Name: "signing_key_pem", Label: "Private key (PEM)", Kind: KindTextarea, Protocol: connections.OIDC, WriteOnly: true,
			Help: "RSA of 2048 bits or more, or EC P-256; sealed at rest."},
	}
	if keyID {
		fs = append(fs, Field{Name: "signing_key_id", Label: "Key ID", Kind: KindText, Protocol: connections.OIDC,
			Help: "Only when the provider assigned the key an ID; Starlight otherwise sends the key's thumbprint."})
	}
	return fs
}
