package oidcfake

import (
	"crypto/sha1"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"
)

// jwtBearer is the RFC 7523 client assertion type.
const jwtBearer = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"

// Assertion is a client assertion a token endpoint accepted.
type Assertion struct {
	Grant    string
	Audience string
	ID       string
	Alg      string
	KeyID    string
}

// RequireCertificate makes every token endpoint refuse secrets and accept only assertions signed by certPEM's key.
func (p *Provider) RequireCertificate(certPEM string) error {
	block, _ := pem.Decode([]byte(certPEM))
	if block == nil {
		return errors.New("oidcfake: certificate is not PEM")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.cert = cert
	p.mu.Unlock()
	return nil
}

// Assertions lists the client assertions the token endpoints accepted, oldest first.
func (p *Provider) Assertions() []Assertion {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]Assertion(nil), p.assertions...)
}

// SetGraphGroups sets the groups, id to display name, that the fake Microsoft Graph lists for a user's object ID.
func (p *Provider) SetGraphGroups(oid string, groups map[string]string) {
	p.mu.Lock()
	p.graph[oid] = groups
	p.mu.Unlock()
}

// authenticate checks the client secret, or once a certificate is registered, a client assertion signed by its key.
func (p *Provider) authenticate(r *http.Request) error {
	p.mu.Lock()
	cert := p.cert
	p.mu.Unlock()
	f := r.PostForm
	id, sec, basic := r.BasicAuth()
	if cert == nil {
		if !basic {
			id, sec = f.Get("client_id"), f.Get("client_secret")
		}
		if id != p.ClientID || sec != p.ClientSecret {
			return errors.New("client secret does not match")
		}
		return nil
	}
	if basic || f.Get("client_secret") != "" {
		return errors.New("this client may not authenticate with a secret")
	}
	if f.Get("client_assertion_type") != jwtBearer {
		return errors.New("client_assertion_type must be jwt-bearer")
	}
	if id := f.Get("client_id"); id != "" && id != p.ClientID {
		return errors.New("client_id does not match the assertion")
	}
	return p.checkAssertion(f.Get("client_assertion"), p.Issuer+r.URL.Path, f.Get("grant_type"), cert)
}

// checkAssertion verifies an assertion the way Microsoft Entra ID does, and remembers its jti so it works once.
func (p *Provider) checkAssertion(raw, audience, grant string, cert *x509.Certificate) error {
	tok, err := jwt.ParseSigned(raw, []jose.SignatureAlgorithm{jose.RS256, jose.ES256})
	if err != nil {
		return fmt.Errorf("assertion: %w", err)
	}
	h := tok.Headers[0]
	s1, s256 := sha1.Sum(cert.Raw), sha256.Sum256(cert.Raw)
	if h.KeyID == "" || h.ExtraHeaders["x5t"] != b64(s1[:]) || h.ExtraHeaders["x5t#S256"] != b64(s256[:]) {
		return errors.New("assertion header does not name the registered certificate by kid, x5t and x5t#S256")
	}
	var c jwt.Claims
	if err := tok.Claims(cert.PublicKey, &c); err != nil {
		return fmt.Errorf("assertion signature: %w", err)
	}
	switch {
	case c.Issuer != p.ClientID || c.Subject != p.ClientID:
		return errors.New("assertion iss and sub must be the client ID")
	case len(c.Audience) != 1 || c.Audience[0] != audience:
		return fmt.Errorf("assertion audience must be %s", audience)
	case c.ID == "":
		return errors.New("assertion has no jti")
	case c.Expiry == nil || c.IssuedAt == nil || c.NotBefore == nil:
		return errors.New("assertion needs exp, iat and nbf")
	case c.Expiry.Time().Sub(c.IssuedAt.Time()) > 5*time.Minute:
		return errors.New("assertion lives longer than five minutes")
	}
	if err := c.ValidateWithLeeway(jwt.Expected{Time: time.Now()}, 30*time.Second); err != nil {
		return fmt.Errorf("assertion is stale: %w", err)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.jtis[c.ID] {
		return errors.New("assertion jti was already used")
	}
	p.jtis[c.ID] = true
	p.assertions = append(p.assertions, Assertion{Grant: grant, Audience: audience, ID: c.ID, Alg: h.Algorithm, KeyID: h.KeyID})
	return nil
}

func (p *Provider) graphGroups(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	ok := p.apps[strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")]
	groups := p.graph[r.PathValue("oid")]
	p.mu.Unlock()
	if !ok {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	ids := make([]string, 0, len(groups))
	for id := range groups {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	value := []map[string]string{}
	for _, id := range ids {
		value = append(value, map[string]string{"id": id, "displayName": groups[id]})
	}
	writeJSON(w, map[string]any{"value": value})
}

func b64(b []byte) string { return base64.RawURLEncoding.EncodeToString(b) }
