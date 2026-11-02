package connections

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"regexp"
	"strings"
)

var (
	slugPattern   = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{1,62}$`)
	domainPattern = regexp.MustCompile(`^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$`)
	guidPattern   = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
	claimPattern  = regexp.MustCompile(`^[A-Za-z0-9_:./#@-]{1,256}$`)
)

// ValidationError explains which field is wrong.
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string { return e.Field + ": " + e.Message }

func invalid(field, format string, args ...any) error {
	return &ValidationError{Field: field, Message: fmt.Sprintf(format, args...)}
}

// NormalizeDomains lowercases, trims and de-duplicates domains, rejecting anything that is not a hostname.
func NormalizeDomains(field string, in []string) ([]string, error) {
	seen := map[string]bool{}
	out := []string{}
	for _, d := range in {
		d = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(d)), ".")
		if d == "" || seen[d] {
			continue
		}
		if !domainPattern.MatchString(d) {
			return nil, invalid(field, "%q is not a domain name", d)
		}
		seen[d] = true
		out = append(out, d)
	}
	return out, nil
}

// Validate checks an input before it is stored and normalises it in place.
func (in *Input) Validate() error {
	in.Slug = strings.TrimSpace(in.Slug)
	in.Name = strings.TrimSpace(in.Name)
	if !slugPattern.MatchString(in.Slug) {
		return invalid("slug", "use 2-63 lowercase letters, digits and hyphens")
	}
	if in.Name == "" || len(in.Name) > 120 {
		return invalid("name", "give the connection a name of up to 120 characters")
	}
	domains, err := NormalizeDomains("domains", in.Domains)
	if err != nil {
		return err
	}
	in.Domains = domains
	c := &in.Config
	if c.Policy.AllowedDomains, err = NormalizeDomains("allowed_domains", c.Policy.AllowedDomains); err != nil {
		return err
	}
	if c.Provider.GoogleDomains, err = NormalizeDomains("google_domains", c.Provider.GoogleDomains); err != nil {
		return err
	}
	if err := validClaims(c.Claims); err != nil {
		return err
	}
	switch in.Protocol {
	case OIDC:
		if c.OIDC == nil {
			return invalid("oidc", "OIDC settings are required")
		}
		c.SAML = nil
		if err := in.validOIDC(); err != nil {
			return err
		}
	case SAML:
		if c.SAML == nil {
			return invalid("saml", "SAML settings are required")
		}
		c.OIDC = nil
		if err := validSAML(c.SAML); err != nil {
			return err
		}
	default:
		return invalid("protocol", "must be oidc or saml")
	}
	if err := in.validGroups(); err != nil {
		return err
	}
	return in.validPreset()
}

func validClaims(m ClaimMap) error {
	for name, v := range map[string]string{"subject": m.Subject, "email": m.Email, "email_verified": m.EmailVerified, "name": m.Name,
		"username": m.Username, "groups": m.Groups, "tenant": m.Tenant} {
		if v != "" && !claimPattern.MatchString(v) {
			return invalid("claims."+name, "%q is not a claim or attribute name", v)
		}
	}
	return nil
}

func (in *Input) validOIDC() error {
	o := in.Config.OIDC
	o.Issuer = strings.TrimSpace(o.Issuer)
	if err := httpsURL("issuer", o.Issuer); err != nil {
		return err
	}
	if strings.TrimSpace(o.ClientID) == "" {
		return invalid("client_id", "is required")
	}
	if o.SigningCertificate = strings.TrimSpace(o.SigningCertificate); o.SigningCertificate != "" {
		cert, err := ParseCertificate(o.SigningCertificate)
		if err != nil {
			return err
		}
		o.SigningCertificate = CertificatePEM(cert)
	}
	switch o.AuthMethod {
	case "":
		o.AuthMethod = AuthBasic
	case AuthBasic, AuthPost, AuthPrivateJWT:
	default:
		return invalid("auth_method", "must be client_secret_basic, client_secret_post or private_key_jwt")
	}
	hasOpenID := false
	for _, s := range o.Scopes {
		if s == "openid" {
			hasOpenID = true
		}
		if s == "" || strings.ContainsAny(s, " \t\r\n") {
			return invalid("scopes", "scopes must be single words")
		}
	}
	if !hasOpenID {
		o.Scopes = append([]string{"openid"}, o.Scopes...)
	}
	for k, v := range o.AuthParams {
		switch k {
		case "client_id", "redirect_uri", "response_type", "scope", "state", "nonce", "code_challenge", "code_challenge_method":
			return invalid("auth_params", "%q is set by Starlight and cannot be overridden", k)
		}
		if strings.ContainsAny(k+v, "\r\n") {
			return invalid("auth_params", "values must be single-line")
		}
	}
	if o.AcceptBearer && len(o.BearerAudiences) == 0 {
		return invalid("bearer_audiences", "name the audience your IdP puts in access tokens for Starlight")
	}
	return nil
}

func validSAML(s *SAMLConfig) error {
	s.MetadataURL = strings.TrimSpace(s.MetadataURL)
	s.MetadataXML = strings.TrimSpace(s.MetadataXML)
	if (s.MetadataURL == "") == (s.MetadataXML == "") {
		return invalid("metadata", "give exactly one of a metadata URL or metadata XML")
	}
	if s.MetadataURL != "" {
		if err := httpsURL("metadata_url", s.MetadataURL); err != nil {
			return err
		}
	}
	if len(s.MetadataXML) > 1<<20 {
		return invalid("metadata_xml", "metadata is larger than 1 MiB")
	}
	if s.SPCertificate = strings.TrimSpace(s.SPCertificate); s.SPCertificate != "" {
		cert, err := parseCertificate("sp_certificate", s.SPCertificate)
		if err != nil {
			return err
		}
		s.SPCertificate = CertificatePEM(cert)
	}
	switch s.NameIDFormat {
	case "", "persistent", "emailAddress", "unspecified", "transient":
	default:
		return invalid("name_id_format", "must be persistent, emailAddress, unspecified or transient")
	}
	return nil
}

func (in *Input) validGroups() error {
	g := &in.Config.Groups
	switch g.Strategy {
	case "":
		g.Strategy = GroupsNone
		if in.Config.Claims.Groups != "" {
			g.Strategy = GroupsClaim
		}
	case GroupsNone, GroupsSCIM:
	case GroupsClaim, GroupsEntraGraph:
		if in.Config.Claims.Groups == "" {
			return invalid("claims.groups", "name the claim or attribute that carries groups")
		}
	case GroupsGoogleDirectory:
		if in.Config.Provider.GoogleAdminEmail == "" || !strings.Contains(in.Config.Provider.GoogleAdminEmail, "@") {
			return invalid("google_admin_email", "name the Workspace admin the service account acts as")
		}
	default:
		return invalid("groups.strategy", "unknown strategy %q", g.Strategy)
	}
	if g.NestedDepth < 0 || g.NestedDepth > 5 {
		return invalid("groups.nested_depth", "must be between 0 and 5")
	}
	return nil
}

func (in *Input) validPreset() error {
	p := &in.Config.Provider
	switch in.Preset {
	case "entra":
		p.EntraTenantID = strings.ToLower(strings.TrimSpace(p.EntraTenantID))
		if !guidPattern.MatchString(p.EntraTenantID) {
			return invalid("entra_tenant_id", "use the directory (tenant) ID, a GUID")
		}
	case "google":
		if len(p.GoogleDomains) == 0 {
			return invalid("google_domains", "list your Workspace domains; without them any Google account could sign in")
		}
	case "gitlab":
		if in.Protocol == OIDC && isGitLabCom(in.Config.OIDC.Issuer) && len(in.Config.Policy.RequiredGroups) == 0 {
			return invalid("required_groups", "gitlab.com is public; require membership of at least one of your groups")
		}
	}
	if in.Config.Groups.Strategy == GroupsEntraGraph && in.Preset != "entra" {
		return invalid("groups.strategy", "Microsoft Graph groups need the Entra ID preset")
	}
	if in.Config.Groups.Strategy == GroupsGoogleDirectory && in.Preset != "google" {
		return invalid("groups.strategy", "Workspace directory groups need the Google preset")
	}
	return nil
}

func isGitLabCom(issuer string) bool {
	u, err := url.Parse(issuer)
	return err == nil && strings.EqualFold(u.Hostname(), "gitlab.com")
}

func httpsURL(field, raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" || u.User != nil || u.Fragment != "" {
		return invalid(field, "must be an absolute URL")
	}
	if u.Scheme == "https" {
		return nil
	}
	if u.Scheme == "http" && isLoopback(u.Hostname()) {
		return nil
	}
	return invalid(field, "must use https")
}

func isLoopback(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// ErrDomainTaken means another connection already routes that email domain.
var ErrDomainTaken = errors.New("connections: another connection already routes one of these domains")
