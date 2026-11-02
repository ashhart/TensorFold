package saml

import (
	"bytes"
	"compress/flate"
	"crypto"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/beevik/etree"
	crewsaml "github.com/crewjam/saml"
	dsig "github.com/russellhaering/goxmldsig"
)

// maxSessionIndexes bounds how many sessions one LogoutRequest may name.
const maxSessionIndexes = 64

type logoutRequestXML struct {
	XMLName      xml.Name              `xml:"urn:oasis:names:tc:SAML:2.0:protocol LogoutRequest"`
	ID           string                `xml:",attr"`
	Version      string                `xml:",attr"`
	IssueInstant crewsaml.RelaxedTime  `xml:",attr"`
	NotOnOrAfter *crewsaml.RelaxedTime `xml:",attr"`
	Destination  string                `xml:",attr"`
	Issuer       *crewsaml.Issuer      `xml:"urn:oasis:names:tc:SAML:2.0:assertion Issuer"`
	NameID       *crewsaml.NameID      `xml:"urn:oasis:names:tc:SAML:2.0:assertion NameID"`
	EncryptedID  *struct{}             `xml:"urn:oasis:names:tc:SAML:2.0:assertion EncryptedID"`
	SessionIndex []string              `xml:"urn:oasis:names:tc:SAML:2.0:protocol SessionIndex"`
}

type logoutResponseXML struct {
	XMLName      xml.Name             `xml:"urn:oasis:names:tc:SAML:2.0:protocol LogoutResponse"`
	ID           string               `xml:",attr"`
	InResponseTo string               `xml:",attr"`
	Version      string               `xml:",attr"`
	IssueInstant crewsaml.RelaxedTime `xml:",attr"`
	Destination  string               `xml:",attr"`
	Issuer       *crewsaml.Issuer     `xml:"urn:oasis:names:tc:SAML:2.0:assertion Issuer"`
	Status       crewsaml.Status      `xml:"urn:oasis:names:tc:SAML:2.0:protocol Status"`
}

// LogoutRequest is a verified IdP-initiated LogoutRequest; Expires is when it stops being acceptable.
type LogoutRequest struct {
	ID             string
	NameID         string
	SessionIndexes []string
	Expires        time.Time
}

// LogoutResponse is a verified answer to a LogoutRequest; the caller matches InResponseTo to a request Starlight sent.
type LogoutResponse struct {
	InResponseTo string
	Status       string
}

// Success reports whether the IdP says it ended the session.
func (r *LogoutResponse) Success() bool { return r.Status == crewsaml.StatusSuccess }

// NameID is the subject of a LogoutRequest, repeated exactly as the IdP asserted it at sign-in.
type NameID struct {
	Value           string
	Format          string
	NameQualifier   string
	SPNameQualifier string
}

// Outbound is a signed logout message for the browser to carry: a redirect URL, or a form to post when Form is set.
type Outbound struct {
	URL  string
	Form url.Values
}

// VerifyLogoutRequest checks an IdP-initiated LogoutRequest's signature, issuer, destination and freshness.
func (s *SP) VerifyLogoutRequest(m Message) (*LogoutRequest, error) {
	if !m.Request {
		return nil, fmt.Errorf("%w: expected a SAMLRequest", ErrLogout)
	}
	el, err := s.verified(m)
	if err != nil {
		return nil, err
	}
	var x logoutRequestXML
	if err := unmarshal(el, &x); err != nil {
		return nil, fmt.Errorf("%w: not a LogoutRequest", ErrLogout)
	}
	issued := time.Time(x.IssueInstant)
	if err := s.envelope(x.Version, x.Issuer, x.Destination, issued); err != nil {
		return nil, err
	}
	expires := issued.Add(logoutMaxAge)
	if x.NotOnOrAfter != nil {
		end := time.Time(*x.NotOnOrAfter)
		if !s.now().Before(end) {
			return nil, fmt.Errorf("%w: request expired", ErrLogout)
		}
		if end.Before(expires) {
			expires = end
		}
	}
	switch {
	case x.ID == "" || len(x.ID) > 256:
		return nil, fmt.Errorf("%w: request has no usable ID", ErrLogout)
	case x.EncryptedID != nil:
		return nil, fmt.Errorf("%w: encrypted NameIDs are not supported", ErrLogout)
	case x.NameID == nil || strings.TrimSpace(x.NameID.Value) == "":
		return nil, fmt.Errorf("%w: request names no NameID", ErrLogout)
	case len(x.SessionIndex) > maxSessionIndexes:
		return nil, fmt.Errorf("%w: too many session indexes", ErrLogout)
	}
	out := &LogoutRequest{ID: x.ID, NameID: strings.TrimSpace(x.NameID.Value), Expires: expires}
	for _, idx := range x.SessionIndex {
		if idx = strings.TrimSpace(idx); idx != "" {
			out.SessionIndexes = append(out.SessionIndexes, idx)
		}
	}
	return out, nil
}

// VerifyLogoutResponse checks a LogoutResponse's signature, issuer, destination and freshness.
func (s *SP) VerifyLogoutResponse(m Message) (*LogoutResponse, error) {
	if m.Request {
		return nil, fmt.Errorf("%w: expected a SAMLResponse", ErrLogout)
	}
	el, err := s.verified(m)
	if err != nil {
		return nil, err
	}
	var x logoutResponseXML
	if err := unmarshal(el, &x); err != nil {
		return nil, fmt.Errorf("%w: not a LogoutResponse", ErrLogout)
	}
	if err := s.envelope(x.Version, x.Issuer, x.Destination, time.Time(x.IssueInstant)); err != nil {
		return nil, err
	}
	if x.InResponseTo == "" {
		return nil, fmt.Errorf("%w: response answers no request", ErrLogout)
	}
	return &LogoutResponse{InResponseTo: x.InResponseTo, Status: x.Status.StatusCode.Value}, nil
}

// HasSLO reports whether the IdP metadata lists a single logout endpoint Starlight can send to.
func (s *SP) HasSLO() bool {
	_, _, ok := s.endpoint(false)
	return ok
}

// endpoint picks the IdP's logout endpoint, preferring HTTP-Redirect; responses use ResponseLocation when set.
func (s *SP) endpoint(response bool) (binding, location string, ok bool) {
	for _, want := range []string{crewsaml.HTTPRedirectBinding, crewsaml.HTTPPostBinding} {
		for _, d := range s.sp.IDPMetadata.IDPSSODescriptors {
			for _, e := range d.SingleLogoutServices {
				loc := e.Location
				if response && e.ResponseLocation != "" {
					loc = e.ResponseLocation
				}
				if u, err := url.Parse(loc); e.Binding == want && err == nil && u.Host != "" && (u.Scheme == "https" || u.Scheme == "http") {
					return want, loc, true
				}
			}
		}
	}
	return "", "", false
}

// LogoutRequest signs a LogoutRequest that ends one IdP session and returns it with the ID the answer must carry.
func (s *SP) LogoutRequest(n NameID, sessionIndex string) (Outbound, string, error) {
	binding, dest, ok := s.endpoint(false)
	if !ok {
		return Outbound{}, "", ErrNoSLO
	}
	if n.Value == "" {
		return Outbound{}, "", errors.New("saml: logout needs the session's NameID")
	}
	req := &crewsaml.LogoutRequest{ID: newID(), Version: "2.0", IssueInstant: s.now().UTC(), Destination: dest,
		Issuer: &crewsaml.Issuer{Format: entityFormat, Value: s.sp.EntityID},
		NameID: &crewsaml.NameID{Value: n.Value, Format: n.Format, NameQualifier: n.NameQualifier, SPNameQualifier: n.SPNameQualifier}}
	if sessionIndex != "" {
		req.SessionIndex = &crewsaml.SessionIndex{Value: sessionIndex}
	}
	out, err := s.outbound(binding, dest, "SAMLRequest", req.Element(), "")
	return out, req.ID, err
}

// LogoutResponse signs the Success answer to an IdP-initiated LogoutRequest, echoing its RelayState.
func (s *SP) LogoutResponse(inResponseTo, relay string) (Outbound, error) {
	binding, dest, ok := s.endpoint(true)
	if !ok {
		return Outbound{}, ErrNoSLO
	}
	resp := &crewsaml.LogoutResponse{ID: newID(), InResponseTo: inResponseTo, Version: "2.0", IssueInstant: s.now().UTC(), Destination: dest,
		Issuer: &crewsaml.Issuer{Format: entityFormat, Value: s.sp.EntityID},
		Status: crewsaml.Status{StatusCode: crewsaml.StatusCode{Value: crewsaml.StatusSuccess}}}
	return s.outbound(binding, dest, "SAMLResponse", resp.Element(), relay)
}

// outbound encodes a message for its binding: a signed query for HTTP-Redirect, an enveloped signature for HTTP-POST.
func (s *SP) outbound(binding, dest, param string, el *etree.Element, relay string) (Outbound, error) {
	if binding == crewsaml.HTTPPostBinding {
		signed, err := s.signEnveloped(el)
		if err != nil {
			return Outbound{}, err
		}
		doc := etree.NewDocument()
		doc.SetRoot(signed)
		b, err := doc.WriteToBytes()
		if err != nil {
			return Outbound{}, err
		}
		form := url.Values{param: {base64.StdEncoding.EncodeToString(b)}}
		if relay != "" {
			form.Set("RelayState", relay)
		}
		return Outbound{URL: dest, Form: form}, nil
	}
	doc := etree.NewDocument()
	doc.SetRoot(el)
	var buf bytes.Buffer
	fw, err := flate.NewWriter(&buf, flate.BestCompression)
	if err != nil {
		return Outbound{}, err
	}
	if _, err := doc.WriteTo(fw); err != nil {
		return Outbound{}, err
	}
	if err := fw.Close(); err != nil {
		return Outbound{}, err
	}
	q := param + "=" + url.QueryEscape(base64.StdEncoding.EncodeToString(buf.Bytes()))
	if relay != "" {
		q += "&RelayState=" + url.QueryEscape(relay)
	}
	q += "&SigAlg=" + url.QueryEscape(dsig.RSASHA256SignatureMethod)
	sum := sha256.Sum256([]byte(q))
	sig, err := s.sp.Key.Sign(rand.Reader, sum[:], crypto.SHA256)
	if err != nil {
		return Outbound{}, err
	}
	q += "&Signature=" + url.QueryEscape(base64.StdEncoding.EncodeToString(sig))
	u, err := url.Parse(dest)
	if err != nil {
		return Outbound{}, err
	}
	if u.RawQuery != "" {
		q = u.RawQuery + "&" + q
	}
	u.RawQuery = q
	return Outbound{URL: u.String()}, nil
}

// signEnveloped signs el and moves the Signature after Issuer; goxmldsig appends it unadopted, so it is sliced off.
func (s *SP) signEnveloped(el *etree.Element) (*etree.Element, error) {
	ctx, err := dsig.NewSigningContext(s.sp.Key, [][]byte{s.sp.Certificate.Raw})
	if err != nil {
		return nil, err
	}
	ctx.Canonicalizer = dsig.MakeC14N10ExclusiveCanonicalizerWithPrefixList("")
	if err := ctx.SetSignatureMethod(dsig.RSASHA256SignatureMethod); err != nil {
		return nil, err
	}
	signed, err := ctx.SignEnveloped(el)
	if err != nil {
		return nil, err
	}
	sig := signed.Child[len(signed.Child)-1]
	signed.Child = signed.Child[:len(signed.Child)-1]
	at := 0
	if issuer := signed.SelectElement("Issuer"); issuer != nil {
		at = issuer.Index() + 1
	}
	signed.InsertChildAt(at, sig)
	return signed, nil
}

func newID() string {
	b := make([]byte, 20)
	_, _ = rand.Read(b)
	return "id-" + hex.EncodeToString(b)
}
