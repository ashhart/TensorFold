// Package mapping turns verified IdP claims into a Starlight user under each connection's policy.
package mapping

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

// Identity is one verified sign-in, normalised across protocols.
type Identity struct {
	Subject       string
	RawSubject    string
	Email         string
	EmailVerified bool
	Name          string
	Username      string
	Groups        []directory.GroupRef
	GroupsKnown   bool
	Overage       bool
	SessionID     string
	AuthTime      time.Time
	AMR           []string
}

// Sign-in refusals; each maps to a message the user can act on.
var (
	ErrMissingSubject   = errors.New("mapping: the identity provider sent no stable subject")
	ErrMissingEmail     = errors.New("mapping: the identity provider sent no email address")
	ErrTenant           = errors.New("mapping: account belongs to another Entra tenant")
	ErrHostedDomain     = errors.New("mapping: account is not in an allowed Workspace domain")
	ErrEmailUnverified  = errors.New("mapping: email is not verified and already belongs to another account")
	ErrIdentityConflict = errors.New("mapping: that account is already linked to a different identity")
	ErrNotProvisioned   = errors.New("mapping: no Starlight account exists and automatic creation is off")
	ErrDisabled         = errors.New("mapping: the Starlight account is suspended or deactivated")
	ErrDomainNotAllowed = errors.New("mapping: email domain is not allowed on this connection")
	ErrGroupRequired    = errors.New("mapping: account is not in a group allowed to sign in")
	ErrUnverified       = errors.New("mapping: the identity provider has not verified this account's email")
)

// FromClaims normalises verified OIDC claims; rawSub is the ID token's sub.
func FromClaims(c connections.Connection, rawSub string, claims map[string]any) (Identity, error) {
	m := c.Config.Claims
	id := Identity{RawSubject: rawSub, SessionID: str(claims["sid"]), AMR: strs(claims["amr"])}
	if at, ok := num(claims["auth_time"]); ok {
		id.AuthTime = time.Unix(int64(at), 0)
	}
	id.Subject = rawSub
	if m.Subject != "" && m.Subject != "sub" {
		id.Subject = str(claims[m.Subject])
	}
	if m.Tenant != "" {
		tid := strings.ToLower(str(claims[m.Tenant]))
		if tid == "" || tid != strings.ToLower(c.Config.Provider.EntraTenantID) {
			return id, ErrTenant
		}
		id.Subject = tid + ":" + strings.ToLower(id.Subject)
	}
	if strings.TrimSpace(id.Subject) == "" || strings.TrimSpace(id.Subject) == ":" {
		return id, ErrMissingSubject
	}
	id.Email = strings.TrimSpace(str(claims[or(m.Email, "email")]))
	id.Username = str(claims[m.Username])
	id.Name = strings.TrimSpace(str(claims[or(m.Name, "name")]))
	verified := truthy(claims[or(m.EmailVerified, "email_verified")])
	switch c.Preset {
	case "entra":
		if id.Email == "" && strings.Contains(id.Username, "@") {
			id.Email = id.Username
		}
		id.EmailVerified = verified || (strings.EqualFold(id.Email, id.Username) && inDomains(id.Email, c.Domains))
		id.Overage = hasOverage(claims)
	case "google":
		hd := strings.ToLower(str(claims["hd"]))
		if hd == "" || !slices.Contains(c.Config.Provider.GoogleDomains, hd) {
			return id, ErrHostedDomain
		}
		if !verified {
			return id, ErrUnverified
		}
		id.EmailVerified = true
	default:
		id.EmailVerified = verified || c.Config.Policy.TrustEmail
	}
	if c.Config.Groups.Strategy == connections.GroupsClaim || c.Config.Groups.Strategy == connections.GroupsEntraGraph {
		id.Groups, id.GroupsKnown = groupRefs(claims[m.Groups], m.GroupsAreKeys), true
	}
	return id, nil
}

// SAMLInput is what a verified SAML assertion contributes.
type SAMLInput struct {
	NameID       string
	NameIDFormat string
	Attributes   map[string][]string
	SessionIndex string
	AuthInstant  time.Time
}

const transientFormat = "urn:oasis:names:tc:SAML:2.0:nameid-format:transient"
const emailFormat = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"

// FromSAML normalises a verified SAML assertion.
func FromSAML(c connections.Connection, in SAMLInput) (Identity, error) {
	m := c.Config.Claims
	first := func(name string) string {
		if v := in.Attributes[name]; len(v) > 0 {
			return strings.TrimSpace(v[0])
		}
		return ""
	}
	id := Identity{RawSubject: in.NameID, SessionID: in.SessionIndex, AuthTime: in.AuthInstant}
	if m.Subject != "" {
		id.Subject = first(m.Subject)
	} else if in.NameIDFormat != transientFormat {
		id.Subject = in.NameID
	}
	if m.Tenant != "" {
		tid := strings.ToLower(first(m.Tenant))
		if tid == "" || tid != strings.ToLower(c.Config.Provider.EntraTenantID) {
			return id, ErrTenant
		}
		id.Subject = tid + ":" + strings.ToLower(id.Subject)
	}
	if strings.TrimSpace(id.Subject) == "" || strings.TrimSpace(id.Subject) == ":" {
		return id, ErrMissingSubject
	}
	id.Email = first(m.Email)
	if id.Email == "" && in.NameIDFormat == emailFormat {
		id.Email = in.NameID
	}
	id.Name = first(m.Name)
	id.Username = first(m.Username)
	id.EmailVerified = c.Config.Policy.TrustEmail
	if c.Preset == "entra" {
		id.EmailVerified = strings.EqualFold(id.Email, id.Username) && inDomains(id.Email, c.Domains)
		id.Overage = len(in.Attributes["http://schemas.microsoft.com/claims/groups.link"]) > 0
	}
	if c.Preset == "google" && !inDomains(id.Email, c.Config.Provider.GoogleDomains) {
		return id, ErrHostedDomain
	}
	if m.Groups != "" && c.Config.Groups.Strategy != connections.GroupsSCIM && c.Config.Groups.Strategy != connections.GroupsNone {
		for _, g := range in.Attributes[m.Groups] {
			if g = strings.TrimSpace(g); g != "" {
				id.Groups = append(id.Groups, directory.GroupRef{ExternalID: g, DisplayName: g})
			}
		}
		id.GroupsKnown = true
	}
	return id, nil
}

func hasOverage(claims map[string]any) bool {
	if names, ok := claims["_claim_names"].(map[string]any); ok {
		if _, ok := names["groups"]; ok {
			return true
		}
	}
	return truthy(claims["hasgroups"])
}

func groupRefs(v any, keys bool) []directory.GroupRef {
	var names []string
	switch g := v.(type) {
	case string:
		names = []string{g}
	case []any:
		for _, x := range g {
			if s, ok := x.(string); ok {
				names = append(names, s)
			} else if x != nil {
				names = append(names, fmt.Sprint(x))
			}
		}
	case map[string]any:
		if keys {
			for k := range g {
				names = append(names, k)
			}
			slices.Sort(names)
		}
	}
	var out []directory.GroupRef
	for _, n := range names {
		if n = strings.TrimSpace(n); n != "" && len(n) <= 512 {
			out = append(out, directory.GroupRef{ExternalID: n, DisplayName: n})
		}
	}
	return out
}

func inDomains(email string, domains []string) bool {
	_, d, ok := strings.Cut(strings.ToLower(email), "@")
	return ok && slices.Contains(domains, d)
}

func truthy(v any) bool {
	switch b := v.(type) {
	case bool:
		return b
	case string:
		return strings.EqualFold(b, "true") || b == "1"
	}
	return false
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func strs(v any) []string {
	var out []string
	if list, ok := v.([]any); ok {
		for _, x := range list {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
	}
	return out
}

func num(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	}
	return 0, false
}

func or(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
