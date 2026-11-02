// Package saml is Starlight's SAML 2.0 service provider: metadata, signed requests, strict checks and single logout.
package saml

import (
	"bytes"
	"context"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	crewsaml "github.com/crewjam/saml"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// Metadata refresh policy for IdPs configured by URL.
const (
	metadataFresh = 24 * time.Hour
	metadataStale = 7 * 24 * time.Hour
	maxMetadata   = 1 << 20
	rsaSHA256     = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"
)

// EntityID is Starlight's SAML entity ID for a connection, which is also its metadata URL.
func EntityID(publicURL, slug string) string {
	return strings.TrimRight(publicURL, "/") + "/auth/" + slug + "/saml/metadata"
}

// ACSURL is where the IdP posts responses for a connection.
func ACSURL(publicURL, slug string) string {
	return strings.TrimRight(publicURL, "/") + "/auth/" + slug + "/saml/acs"
}

// Factory builds and caches service providers per connection.
type Factory struct {
	pool      *pgxpool.Pool
	http      *http.Client
	publicURL string
	now       func() time.Time
	mu        sync.Mutex
	cache     map[string]cachedSP
}

type cachedSP struct {
	updated time.Time
	loaded  time.Time
	sp      *SP
}

// NewFactory returns a factory that fetches IdP metadata with client.
func NewFactory(pool *pgxpool.Pool, client *http.Client, publicURL string, now func() time.Time) *Factory {
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	if now == nil {
		now = time.Now
	}
	return &Factory{pool: pool, http: client, publicURL: strings.TrimRight(publicURL, "/"), now: now, cache: map[string]cachedSP{}}
}

// SP is a configured service provider for one connection.
type SP struct {
	conn connections.Connection
	sp   *crewsaml.ServiceProvider
	now  func() time.Time
}

// SP returns the service provider for c, refreshing URL-sourced IdP metadata at most daily.
func (f *Factory) SP(ctx context.Context, c connections.Connection, secrets connections.Secrets) (*SP, error) {
	f.mu.Lock()
	hit, ok := f.cache[c.ID]
	f.mu.Unlock()
	if ok && hit.updated.Equal(c.UpdatedAt) && f.now().Sub(hit.loaded) < metadataFresh {
		return hit.sp, nil
	}
	s := c.Config.SAML
	if s == nil {
		return nil, errors.New("saml: connection has no SAML settings")
	}
	md, err := f.idpMetadata(ctx, c)
	if err != nil {
		return nil, err
	}
	pair, err := tls.X509KeyPair([]byte(s.SPCertificate), []byte(secrets.SAMLKeyPEM))
	if err != nil {
		return nil, fmt.Errorf("saml: service provider key pair: %w", err)
	}
	key, ok := pair.PrivateKey.(*rsa.PrivateKey)
	if !ok {
		return nil, errors.New("saml: service provider key must be RSA")
	}
	cert, err := x509.ParseCertificate(pair.Certificate[0])
	if err != nil {
		return nil, err
	}
	entity, _ := url.Parse(EntityID(f.publicURL, c.Slug))
	acs, _ := url.Parse(ACSURL(f.publicURL, c.Slug))
	slo, _ := url.Parse(SLOURL(f.publicURL, c.Slug))
	sp := &crewsaml.ServiceProvider{
		EntityID: entity.String(), Key: key, Certificate: cert, MetadataURL: *entity, AcsURL: *acs, SloURL: *slo, IDPMetadata: md,
		AuthnNameIDFormat: nameIDFormat(s.NameIDFormat), AllowIDPInitiated: s.AllowIDPInitiated, HTTPClient: f.http,
		MetadataValidDuration: 7 * 24 * time.Hour, LogoutBindings: []string{crewsaml.HTTPRedirectBinding, crewsaml.HTTPPostBinding},
		SignatureVerifier: shapeVerifier{},
	}
	if s.SignRequests {
		sp.SignatureMethod = rsaSHA256
	}
	sp.ValidateAudienceRestriction = strictAudience(sp.EntityID)
	out := &SP{conn: c, sp: sp, now: f.now}
	f.mu.Lock()
	f.cache[c.ID] = cachedSP{updated: c.UpdatedAt, loaded: f.now(), sp: out}
	f.mu.Unlock()
	return out, nil
}

// Forget drops a cached service provider after its connection changes.
func (f *Factory) Forget(connectionID string) {
	f.mu.Lock()
	delete(f.cache, connectionID)
	f.mu.Unlock()
}

func nameIDFormat(s string) crewsaml.NameIDFormat {
	switch s {
	case "persistent":
		return crewsaml.PersistentNameIDFormat
	case "emailAddress":
		return crewsaml.EmailAddressNameIDFormat
	case "transient":
		return crewsaml.TransientNameIDFormat
	default:
		return crewsaml.UnspecifiedNameIDFormat
	}
}

// strictAudience requires an AudienceRestriction naming Starlight; crewjam alone accepts assertions with none.
func strictAudience(entity string) func(*crewsaml.Assertion) error {
	return func(a *crewsaml.Assertion) error {
		if a.Conditions == nil || len(a.Conditions.AudienceRestrictions) == 0 {
			return errors.New("assertion has no audience restriction")
		}
		for _, r := range a.Conditions.AudienceRestrictions {
			if r.Audience.Value == entity {
				return nil
			}
		}
		return fmt.Errorf("assertion audience does not include %s", entity)
	}
}

func (f *Factory) idpMetadata(ctx context.Context, c connections.Connection) (*crewsaml.EntityDescriptor, error) {
	s := c.Config.SAML
	if s.MetadataXML != "" {
		return ParseMetadata([]byte(s.MetadataXML))
	}
	var body string
	var fetched time.Time
	err := f.pool.QueryRow(ctx, "SELECT body, fetched_at FROM idp_metadata WHERE connection_id = $1", c.ID).Scan(&body, &fetched)
	if err != nil && !db.NotFound(err) {
		return nil, err
	}
	if body != "" && f.now().Sub(fetched) < metadataFresh {
		return ParseMetadata([]byte(body))
	}
	raw, ferr := f.fetch(ctx, s.MetadataURL)
	if ferr == nil {
		md, perr := ParseMetadata(raw)
		if perr == nil {
			_, err := f.pool.Exec(ctx, `INSERT INTO idp_metadata (connection_id, body, fetched_at) VALUES ($1, $2, now())
				ON CONFLICT (connection_id) DO UPDATE SET body = EXCLUDED.body, fetched_at = EXCLUDED.fetched_at`, c.ID, string(raw))
			return md, err
		}
		ferr = perr
	}
	if body != "" && f.now().Sub(fetched) < metadataStale {
		slog.Warn("saml metadata refresh failed; using cached copy", "connection", c.Slug, "err", ferr)
		return ParseMetadata([]byte(body))
	}
	return nil, fmt.Errorf("saml: IdP metadata: %w", ferr)
}

func (f *Factory) fetch(ctx context.Context, target string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, err
	}
	resp, err := f.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metadata URL returned %s", resp.Status)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxMetadata+1))
	if err != nil {
		return nil, err
	}
	if len(raw) > maxMetadata {
		return nil, errors.New("metadata is larger than 1 MiB")
	}
	return raw, nil
}

// ParseMetadata reads IdP metadata, accepting a bare EntityDescriptor or the first IdP inside an EntitiesDescriptor.
func ParseMetadata(raw []byte) (*crewsaml.EntityDescriptor, error) {
	if len(raw) > maxMetadata {
		return nil, errors.New("saml: metadata is larger than 1 MiB")
	}
	var one crewsaml.EntityDescriptor
	if err := xml.Unmarshal(raw, &one); err == nil && len(one.IDPSSODescriptors) > 0 {
		return &one, nil
	}
	var many crewsaml.EntitiesDescriptor
	if err := xml.NewDecoder(bytes.NewReader(raw)).Decode(&many); err == nil {
		for i := range many.EntityDescriptors {
			if len(many.EntityDescriptors[i].IDPSSODescriptors) > 0 {
				return &many.EntityDescriptors[i], nil
			}
		}
	}
	return nil, errors.New("saml: metadata has no IdP SSO descriptor")
}

// Metadata is Starlight's SP metadata: a POST-only consumer, the logout endpoint and the key signing logout messages.
func (s *SP) Metadata() ([]byte, error) {
	md := s.sp.Metadata()
	for i := range md.SPSSODescriptors {
		var post []crewsaml.IndexedEndpoint
		for _, acs := range md.SPSSODescriptors[i].AssertionConsumerServices {
			if acs.Binding == crewsaml.HTTPPostBinding {
				post = append(post, acs)
			}
		}
		md.SPSSODescriptors[i].AssertionConsumerServices = post
	}
	if s.sp.SignatureMethod == "" {
		signing := crewsaml.KeyDescriptor{Use: "signing", KeyInfo: crewsaml.KeyInfo{X509Data: crewsaml.X509Data{
			X509Certificates: []crewsaml.X509Certificate{{Data: base64.StdEncoding.EncodeToString(s.sp.Certificate.Raw)}}}}}
		for i := range md.SPSSODescriptors {
			md.SPSSODescriptors[i].KeyDescriptors = append(md.SPSSODescriptors[i].KeyDescriptors, signing)
		}
	}
	b, err := xml.MarshalIndent(md, "", "  ")
	if err != nil {
		return nil, err
	}
	return append([]byte(xml.Header), b...), nil
}

// AuthnRedirect builds a signed HTTP-Redirect AuthnRequest and returns its URL and request ID.
func (s *SP) AuthnRedirect(relayState string) (string, string, error) {
	dest := s.sp.GetSSOBindingLocation(crewsaml.HTTPRedirectBinding)
	if dest == "" {
		return "", "", errors.New("saml: IdP metadata has no HTTP-Redirect SSO endpoint")
	}
	req, err := s.sp.MakeAuthenticationRequest(dest, crewsaml.HTTPRedirectBinding, crewsaml.HTTPPostBinding)
	if err != nil {
		return "", "", err
	}
	u, err := req.Redirect(relayState, s.sp)
	if err != nil {
		return "", "", err
	}
	return u.String(), req.ID, nil
}
