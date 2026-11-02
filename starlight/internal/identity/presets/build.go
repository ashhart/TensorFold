package presets

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// Values are form inputs keyed by field name.
type Values map[string]string

// FromForm copies the posted values of every field the preset uses.
func FromForm(form url.Values) Values {
	v := Values{}
	for k, vs := range form {
		if len(vs) > 0 {
			v[k] = strings.TrimSpace(vs[0])
		}
	}
	return v
}

// List splits a comma- or newline-separated field.
func (v Values) List(name string) []string {
	var out []string
	for _, part := range strings.FieldsFunc(v[name], func(r rune) bool { return r == ',' || r == '\n' || r == '\r' }) {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// Bool reads a checkbox.
func (v Values) Bool(name string) bool {
	switch strings.ToLower(v[name]) {
	case "on", "true", "1", "yes":
		return true
	}
	return false
}

func (v Values) need(name, label string) (string, error) {
	s := strings.TrimSpace(v[name])
	if s == "" {
		return "", &connections.ValidationError{Field: name, Message: label + " is required"}
	}
	return s, nil
}

// host strips a scheme and trailing slash so admins can paste either a domain or a URL.
func (v Values) host(name, label string) (string, error) {
	s, err := v.need(name, label)
	if err != nil {
		return "", err
	}
	s = strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(s, "https://"), "http://"), "/")
	if strings.ContainsAny(s, "/?#@ ") {
		return "", &connections.ValidationError{Field: name, Message: label + " must be a host name"}
	}
	return strings.ToLower(s), nil
}

// base returns an https base URL without a trailing slash.
func (v Values) base(name, label string) (string, error) {
	s, err := v.need(name, label)
	if err != nil {
		return "", err
	}
	if !strings.Contains(s, "://") {
		s = "https://" + s
	}
	return strings.TrimRight(s, "/"), nil
}

// Build turns form values into a connection input for this preset and protocol.
func (p Preset) Build(protocol string, v Values) (connections.Input, error) {
	if !p.Supports(protocol) {
		return connections.Input{}, fmt.Errorf("%s does not support %s", p.Label, protocol)
	}
	in := connections.Input{
		Slug: v["slug"], Name: v["name"], Protocol: protocol, Preset: p.Name, Domains: v.List("domains"),
		Enabled: true, JIT: v.Bool("jit"),
	}
	cfg := &in.Config
	cfg.Policy.RequiredGroups = v.List("required_groups")
	cfg.Policy.AllowedDomains = v.List("allowed_domains")
	if protocol == connections.OIDC {
		issuer := v["issuer"]
		if p.issuer != nil {
			var err error
			if issuer, err = p.issuer(v); err != nil {
				return in, err
			}
		}
		cfg.OIDC = &connections.OIDCConfig{Issuer: issuer, ClientID: v["client_id"], AuthMethod: v["auth_method"],
			Scopes: []string{"openid", "email", "profile"}, UseUserinfo: v.Bool("use_userinfo"), RPLogout: v.Bool("rp_logout"),
			AcceptBearer: v.Bool("accept_bearer"), BearerAudiences: v.List("bearer_audiences"), SigningKeyID: v["signing_key_id"],
			SigningCertificate: v["signing_certificate"]}
		in.Secrets.ClientSecret = v["client_secret"]
		in.Secrets.SigningKeyPEM = v["signing_key_pem"]
		if v.Bool("generate_certificate") {
			if err := generateClientKey(&in); err != nil {
				return in, err
			}
		}
		cfg.Policy.TrustEmail = v.Bool("trust_email")
		cfg.Claims = connections.ClaimMap{Email: "email", EmailVerified: "email_verified", Name: "name", Username: "preferred_username"}
		if p.oidc != nil {
			p.oidc(&in, v)
		}
		if extra := v.List("scopes"); len(extra) > 0 {
			cfg.OIDC.Scopes = append([]string{"openid"}, without(extra, "openid")...)
		}
	} else {
		cfg.SAML = &connections.SAMLConfig{MetadataURL: v["metadata_url"], MetadataXML: v["metadata_xml"], SignRequests: true,
			NameIDFormat: v["name_id_format"], AllowIDPInitiated: v.Bool("allow_idp_initiated"), RPLogout: v.Bool("rp_logout")}
		cfg.Claims = connections.ClaimMap{Email: "email", Name: "name"}
		cfg.Policy.TrustEmail = true
		if p.saml != nil {
			p.saml(&in, v)
		}
	}
	overrideClaims(cfg, v)
	if cfg.Groups.Strategy == "" && cfg.Claims.Groups != "" {
		cfg.Groups.Strategy = connections.GroupsClaim
	}
	return in, nil
}

// generateClientKey makes the private_key_jwt key and certificate, refusing to mix them with pasted ones.
func generateClientKey(in *connections.Input) error {
	if in.Secrets.SigningKeyPEM != "" || in.Config.OIDC.SigningCertificate != "" {
		return &connections.ValidationError{Field: "generate_certificate", Message: "paste a key and certificate, or generate them, not both"}
	}
	cert, key, err := connections.NewClientKeyPair("starlight-" + in.Slug)
	if err != nil {
		return err
	}
	in.Config.OIDC.SigningCertificate, in.Secrets.SigningKeyPEM = cert, key
	return nil
}

func overrideClaims(cfg *connections.Config, v Values) {
	set := func(dst *string, key string) {
		if s := strings.TrimSpace(v[key]); s != "" {
			*dst = s
		}
	}
	set(&cfg.Claims.Subject, "subject_claim")
	set(&cfg.Claims.Email, "email_claim")
	set(&cfg.Claims.Name, "name_claim")
	set(&cfg.Claims.Groups, "groups_claim")
}

func without(in []string, drop string) []string {
	out := in[:0:0]
	for _, s := range in {
		if s != drop {
			out = append(out, s)
		}
	}
	return out
}

func both() []string     { return []string{connections.OIDC, connections.SAML} }
func oidcOnly() []string { return []string{connections.OIDC} }
func samlOnly() []string { return []string{connections.SAML} }

func fixed(issuer string) func(Values) (string, error) {
	return func(Values) (string, error) { return issuer, nil }
}

func groupsClaim(name string) func(*connections.Input, Values) {
	return func(in *connections.Input, _ Values) { in.Config.Claims.Groups = name }
}
