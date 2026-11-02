// Package samlfake is an in-process SAML identity provider, built on crewjam/saml, for end-to-end tests.
package samlfake

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/xml"
	"errors"
	"html"
	"log"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"regexp"
	"sync"
	"time"

	crewsaml "github.com/crewjam/saml"
	dsig "github.com/russellhaering/goxmldsig"
)

// IdP is a TLS test identity provider that signs assertions for one registered service provider and speaks SLO.
type IdP struct {
	Server *httptest.Server
	IDP    *crewsaml.IdentityProvider

	mu          sync.Mutex
	sp          *crewsaml.EntityDescriptor
	session     crewsaml.Session
	mutate      func(*crewsaml.Assertion)
	sloBindings []string
	reply       Logout
	requests    []Seen
	responses   []Seen
}

// New starts an IdP with a fresh RSA signing key.
func New() *IdP {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic(err)
	}
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "test idp"}, NotBefore: time.Now().Add(-time.Hour),
		NotAfter: time.Now().Add(24 * time.Hour), KeyUsage: x509.KeyUsageDigitalSignature}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		panic(err)
	}
	cert, _ := x509.ParseCertificate(der)
	i := &IdP{sloBindings: []string{crewsaml.HTTPRedirectBinding, crewsaml.HTTPPostBinding}}
	mux := http.NewServeMux()
	i.Server = httptest.NewTLSServer(mux)
	base, _ := url.Parse(i.Server.URL)
	md, sso := *base, *base
	md.Path, sso.Path = "/metadata", "/sso"
	i.IDP = &crewsaml.IdentityProvider{Key: key, Certificate: cert, Logger: log.New(os.Stderr, "samlfake ", 0), MetadataURL: md, SSOURL: sso,
		ServiceProviderProvider: i, SessionProvider: i, AssertionMaker: i, SignatureMethod: dsig.RSASHA256SignatureMethod}
	mux.HandleFunc("/metadata", i.IDP.ServeMetadata)
	mux.HandleFunc("/sso", i.IDP.ServeSSO)
	mux.HandleFunc("/initiated", func(w http.ResponseWriter, r *http.Request) {
		i.IDP.ServeIDPInitiated(w, r, r.URL.Query().Get("sp"), r.URL.Query().Get("relay"))
	})
	mux.HandleFunc("/slo", i.serveSLO)
	return i
}

// Close stops the server.
func (i *IdP) Close() { i.Server.Close() }

// MetadataXML is the IdP's metadata document, with a single logout endpoint for each binding SetSLOBindings left.
func (i *IdP) MetadataXML() string {
	md := i.IDP.Metadata()
	i.mu.Lock()
	md.IDPSSODescriptors[0].SingleLogoutServices = nil
	for _, b := range i.sloBindings {
		md.IDPSSODescriptors[0].SingleLogoutServices = append(md.IDPSSODescriptors[0].SingleLogoutServices,
			crewsaml.Endpoint{Binding: b, Location: i.sloURL()})
	}
	i.mu.Unlock()
	b, _ := xml.MarshalIndent(md, "", " ")
	return string(b)
}

// RegisterSP trusts the service provider described by metadata.
func (i *IdP) RegisterSP(metadata []byte) error {
	var ed crewsaml.EntityDescriptor
	if err := xml.Unmarshal(metadata, &ed); err != nil {
		return err
	}
	i.mu.Lock()
	i.sp = &ed
	i.mu.Unlock()
	return nil
}

// SetSession sets who the next assertion is about.
func (i *IdP) SetSession(s crewsaml.Session) {
	i.mu.Lock()
	i.session = s
	i.mu.Unlock()
}

// Mutate edits each assertion before it is signed, to forge bad responses.
func (i *IdP) Mutate(f func(*crewsaml.Assertion)) {
	i.mu.Lock()
	i.mutate = f
	i.mu.Unlock()
}

// GetServiceProvider implements crewsaml.ServiceProviderProvider.
func (i *IdP) GetServiceProvider(_ *http.Request, id string) (*crewsaml.EntityDescriptor, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.sp == nil || i.sp.EntityID != id {
		return nil, os.ErrNotExist
	}
	return i.sp, nil
}

// GetSession implements crewsaml.SessionProvider.
func (i *IdP) GetSession(http.ResponseWriter, *http.Request, *crewsaml.IdpAuthnRequest) *crewsaml.Session {
	i.mu.Lock()
	defer i.mu.Unlock()
	s := i.session
	if s.ID == "" {
		s.ID = "session-1"
	}
	if s.CreateTime.IsZero() {
		s.CreateTime = time.Now()
	}
	if s.ExpireTime.IsZero() {
		s.ExpireTime = time.Now().Add(time.Hour)
	}
	return &s
}

// MakeAssertion implements crewsaml.AssertionMaker, applying Mutate after the default assertion.
func (i *IdP) MakeAssertion(req *crewsaml.IdpAuthnRequest, s *crewsaml.Session) error {
	if err := (crewsaml.DefaultAssertionMaker{}).MakeAssertion(req, s); err != nil {
		return err
	}
	i.mu.Lock()
	f := i.mutate
	i.mu.Unlock()
	if f != nil {
		f(req.Assertion)
	}
	return nil
}

var (
	actionRe = regexp.MustCompile(`<form[^>]+action="([^"]+)"`)
	fieldRe  = regexp.MustCompile(`<input[^>]+name="([A-Za-z]+)"[^>]+value="([^"]*)"`)
)

// Form extracts the auto-post form an IdP returns: target URL, SAMLResponse and RelayState.
func Form(body []byte) (action, response, relay string, err error) {
	m := actionRe.FindSubmatch(body)
	if m == nil {
		return "", "", "", errors.New("samlfake: no form in IdP response")
	}
	action = html.UnescapeString(string(m[1]))
	for _, f := range fieldRe.FindAllSubmatch(body, -1) {
		switch string(f[1]) {
		case "SAMLResponse":
			response = html.UnescapeString(string(f[2]))
		case "RelayState":
			relay = html.UnescapeString(string(f[2]))
		}
	}
	if response == "" {
		return "", "", "", errors.New("samlfake: no SAMLResponse in form")
	}
	return action, response, relay, nil
}
