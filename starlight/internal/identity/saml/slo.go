package saml

import (
	"bytes"
	"compress/flate"
	"crypto/x509"
	"encoding/base64"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/beevik/etree"
	crewsaml "github.com/crewjam/saml"
	xrv "github.com/mattermost/xml-roundtrip-validator"
	dsig "github.com/russellhaering/goxmldsig"
)

// SLO limits: a message's size as sent and decoded, and how far its IssueInstant may sit from Starlight's clock.
const (
	maxLogout    = 32 << 10
	logoutSkew   = 3 * time.Minute
	logoutMaxAge = 10 * time.Minute
	entityFormat = "urn:oasis:names:tc:SAML:2.0:nameid-format:entity"
)

// ErrLogout wraps every refused logout message; the detail is for logs and the audit trail.
var ErrLogout = errors.New("saml: logout message rejected")

// ErrNoSLO means the IdP metadata lists no single logout endpoint with a binding Starlight speaks.
var ErrNoSLO = errors.New("saml: IdP metadata has no single logout endpoint")

// redirectAlgs are the HTTP-Redirect signature algorithms Starlight accepts; SHA-1 is refused.
var redirectAlgs = map[string]x509.SignatureAlgorithm{
	dsig.RSASHA256SignatureMethod: x509.SHA256WithRSA,
	dsig.RSASHA384SignatureMethod: x509.SHA384WithRSA,
	dsig.RSASHA512SignatureMethod: x509.SHA512WithRSA,
}

var whitespace = regexp.MustCompile(`\s+`)

// SLOURL is Starlight's single logout endpoint for a connection, for requests and responses over both bindings.
func SLOURL(publicURL, slug string) string {
	return strings.TrimRight(publicURL, "/") + "/auth/" + slug + "/saml/slo"
}

// Message is a SAMLRequest or SAMLResponse as the browser delivered it, before any check.
type Message struct {
	Request    bool
	Redirect   bool
	RelayState string
	payload    string
	sigAlg     string
	signature  string
	signed     string
}

// Binding names the message's binding for logs and audit records.
func (m Message) Binding() string {
	if m.Redirect {
		return "redirect"
	}
	return "post"
}

// ReadMessage takes a logout message from an HTTP-Redirect query or an HTTP-POST form; the caller limits the body.
func ReadMessage(r *http.Request) (Message, error) {
	if r.Method == http.MethodGet {
		return readRedirect(r.URL.RawQuery)
	}
	if err := r.ParseForm(); err != nil {
		return Message{}, fmt.Errorf("%w: unreadable form", ErrLogout)
	}
	req, resp := r.PostForm.Get("SAMLRequest"), r.PostForm.Get("SAMLResponse")
	if (req == "") == (resp == "") {
		return Message{}, fmt.Errorf("%w: send exactly one of SAMLRequest and SAMLResponse", ErrLogout)
	}
	return Message{Request: req != "", RelayState: r.PostForm.Get("RelayState"), payload: req + resp}, nil
}

// readRedirect keeps each parameter as it was encoded, because the query signature covers those exact bytes.
func readRedirect(rawQuery string) (Message, error) {
	raw := map[string]string{}
	for _, part := range strings.Split(rawQuery, "&") {
		if part == "" {
			continue
		}
		k, v, _ := strings.Cut(part, "=")
		key, err := url.QueryUnescape(k)
		if err != nil {
			return Message{}, fmt.Errorf("%w: query is not URL-encoded", ErrLogout)
		}
		if _, dup := raw[key]; dup {
			return Message{}, fmt.Errorf("%w: query repeats %s", ErrLogout, key)
		}
		raw[key] = v
	}
	_, isReq := raw["SAMLRequest"]
	_, isResp := raw["SAMLResponse"]
	if isReq == isResp {
		return Message{}, fmt.Errorf("%w: send exactly one of SAMLRequest and SAMLResponse", ErrLogout)
	}
	m, param := Message{Request: isReq, Redirect: true}, "SAMLResponse"
	if isReq {
		param = "SAMLRequest"
	}
	for k, dst := range map[string]*string{param: &m.payload, "RelayState": &m.RelayState, "SigAlg": &m.sigAlg, "Signature": &m.signature} {
		v, err := url.QueryUnescape(raw[k])
		if err != nil {
			return Message{}, fmt.Errorf("%w: %s is not URL-encoded", ErrLogout, k)
		}
		*dst = v
	}
	if _, ok := raw["SigAlg"]; ok {
		m.signed = param + "=" + raw[param]
		if relay, ok := raw["RelayState"]; ok {
			m.signed += "&RelayState=" + relay
		}
		m.signed += "&SigAlg=" + raw["SigAlg"]
	}
	return m, nil
}

// decode returns the message XML: base64 for HTTP-POST, then raw DEFLATE as well for HTTP-Redirect.
func (m Message) decode() ([]byte, error) {
	if len(m.payload) > maxLogout {
		return nil, fmt.Errorf("%w: message too large", ErrLogout)
	}
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(m.payload))
	if err != nil {
		return nil, fmt.Errorf("%w: not base64", ErrLogout)
	}
	if m.Redirect {
		if raw, err = io.ReadAll(io.LimitReader(flate.NewReader(bytes.NewReader(raw)), maxLogout+1)); err != nil {
			return nil, fmt.Errorf("%w: not DEFLATE-compressed", ErrLogout)
		}
	}
	if len(raw) > maxLogout {
		return nil, fmt.Errorf("%w: message too large", ErrLogout)
	}
	if err := xrv.Validate(bytes.NewReader(raw)); err != nil {
		return nil, fmt.Errorf("%w: malformed XML", ErrLogout)
	}
	return raw, nil
}

// verified checks the message's signature against the IdP metadata certificates and returns the signed XML.
func (s *SP) verified(m Message) (*etree.Element, error) {
	raw, err := m.decode()
	if err != nil {
		return nil, err
	}
	certs, err := signingCerts(s.sp.IDPMetadata, s.now())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrLogout, err)
	}
	if !m.Redirect {
		return verifyEnveloped(raw, certs)
	}
	if m.sigAlg == "" {
		return nil, fmt.Errorf("%w: message is not signed", ErrLogout)
	}
	if err := verifyQuery(m, certs); err != nil {
		return nil, err
	}
	doc := etree.NewDocument()
	if err := doc.ReadFromBytes(raw); err != nil || doc.Root() == nil {
		return nil, fmt.Errorf("%w: malformed XML", ErrLogout)
	}
	return doc.Root(), nil
}

// signingCerts lists the IdP metadata's current signing certificates.
func signingCerts(md *crewsaml.EntityDescriptor, now time.Time) ([]*x509.Certificate, error) {
	var out []*x509.Certificate
	for _, d := range md.IDPSSODescriptors {
		for _, k := range d.KeyDescriptors {
			if k.Use != "" && k.Use != "signing" {
				continue
			}
			for _, c := range k.KeyInfo.X509Data.X509Certificates {
				der, err := base64.StdEncoding.DecodeString(whitespace.ReplaceAllString(c.Data, ""))
				if err != nil {
					continue
				}
				if cert, err := x509.ParseCertificate(der); err == nil && !now.Before(cert.NotBefore) && !now.After(cert.NotAfter) {
					out = append(out, cert)
				}
			}
		}
	}
	if len(out) == 0 {
		return nil, errors.New("IdP metadata has no current signing certificate")
	}
	return out, nil
}

func verifyQuery(m Message, certs []*x509.Certificate) error {
	alg, ok := redirectAlgs[m.sigAlg]
	if !ok {
		return fmt.Errorf("%w: signature algorithm %q is not accepted", ErrLogout, m.sigAlg)
	}
	sig, err := base64.StdEncoding.DecodeString(m.signature)
	if err != nil || len(sig) == 0 {
		return fmt.Errorf("%w: signature is not base64", ErrLogout)
	}
	for _, c := range certs {
		if c.CheckSignature(alg, []byte(m.signed), sig) == nil {
			return nil
		}
	}
	return fmt.Errorf("%w: signature does not match the IdP's certificates", ErrLogout)
}

// verifyEnveloped validates the one enveloped signature and returns the element it covers, never the unsigned original.
func verifyEnveloped(raw []byte, certs []*x509.Certificate) (*etree.Element, error) {
	doc := etree.NewDocument()
	if err := doc.ReadFromBytes(raw); err != nil || doc.Root() == nil {
		return nil, fmt.Errorf("%w: malformed XML", ErrLogout)
	}
	root := doc.Root()
	sigs, err := signatures(root)
	switch {
	case err != nil:
		return nil, fmt.Errorf("%w: %v", ErrLogout, err)
	case len(sigs) == 0:
		return nil, fmt.Errorf("%w: message is not signed", ErrLogout)
	case len(sigs) > 1 || sigs[0].Parent() != root:
		return nil, fmt.Errorf("%w: %v; a logout message carries one, directly under its root", ErrLogout, errMisplaced)
	}
	sig := sigs[0]
	if alg := weakAlgorithm(sig); alg != "" {
		return nil, fmt.Errorf("%w: signature algorithm %q is not accepted", ErrLogout, alg)
	}
	// A signature that carries its certificate is checked once against them all; one without is tried against each.
	stores := [][]*x509.Certificate{certs}
	if sig.FindElement("./KeyInfo/X509Data/X509Certificate") == nil {
		if ki := sig.SelectElement("KeyInfo"); ki != nil {
			sig.RemoveChild(ki)
		}
		stores = stores[:0]
		for _, c := range certs {
			stores = append(stores, []*x509.Certificate{c})
		}
	}
	var last error
	for _, roots := range stores {
		vc := dsig.NewDefaultValidationContext(&dsig.MemoryX509CertificateStore{Roots: roots})
		vc.IdAttribute = "ID"
		el, err := vc.Validate(root)
		if err == nil {
			return el, nil
		}
		last = err
	}
	return nil, fmt.Errorf("%w: signature: %v", ErrLogout, last)
}

func unmarshal(el *etree.Element, v any) error {
	doc := etree.NewDocument()
	doc.SetRoot(el.Copy())
	b, err := doc.WriteToBytes()
	if err != nil {
		return err
	}
	return xml.Unmarshal(b, v)
}

// envelope checks each logout message's SAML 2.0 version, the IdP as issuer, Starlight as destination and a fresh time.
func (s *SP) envelope(version string, issuer *crewsaml.Issuer, destination string, issued time.Time) error {
	switch {
	case version != "2.0":
		return fmt.Errorf("%w: version %q", ErrLogout, version)
	case issuer == nil || strings.TrimSpace(issuer.Value) != s.sp.IDPMetadata.EntityID:
		return fmt.Errorf("%w: issuer is not the IdP in the metadata", ErrLogout)
	case destination != s.sp.SloURL.String():
		return fmt.Errorf("%w: destination is not Starlight's logout URL", ErrLogout)
	}
	now := s.now()
	if issued.IsZero() || issued.After(now.Add(logoutSkew)) || now.Sub(issued) > logoutMaxAge {
		return fmt.Errorf("%w: issued outside the accepted window", ErrLogout)
	}
	return nil
}
