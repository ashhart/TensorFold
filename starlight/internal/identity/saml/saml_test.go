package saml_test

import (
	"context"
	"crypto/x509"
	"encoding/base64"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/beevik/etree"
	crewsaml "github.com/crewjam/saml"
	"github.com/crewjam/saml/xmlenc"
	dsig "github.com/russellhaering/goxmldsig"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
	"github.com/ashhart/TensorFold/starlight/internal/identity/saml/samlfake"
)

type fixture struct {
	idp *samlfake.IdP
	sp  *samlsp.SP
}

func newFixture(t *testing.T, allowIDPInitiated bool) fixture {
	t.Helper()
	idp := samlfake.New()
	t.Cleanup(idp.Close)
	return build(t, idp, func(s *connections.SAMLConfig) { s.AllowIDPInitiated = allowIDPInitiated })
}

// build makes Starlight's SP for idp with edited settings, registers it with the IdP, and gives the IdP a session.
func build(t *testing.T, idp *samlfake.IdP, edit func(*connections.SAMLConfig)) fixture {
	t.Helper()
	cert, key, err := connections.NewSPKeyPair("starlight-sp-corp")
	if err != nil {
		t.Fatal(err)
	}
	s := &connections.SAMLConfig{MetadataXML: idp.MetadataXML(), SignRequests: true, SPCertificate: cert, NameIDFormat: "persistent"}
	edit(s)
	c := connections.Connection{ID: "c1", Slug: "corp", Protocol: connections.SAML, Enabled: true, UpdatedAt: time.Unix(1, 0),
		Config: connections.Config{SAML: s}}
	f := samlsp.NewFactory(nil, idp.Server.Client(), "https://starlight.example", nil)
	sp, err := f.SP(context.Background(), c, connections.Secrets{SAMLKeyPEM: key})
	if err != nil {
		t.Fatal(err)
	}
	md, err := sp.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := idp.RegisterSP(md); err != nil {
		t.Fatal(err)
	}
	idp.SetSession(crewsaml.Session{NameID: "persistent-ada", NameIDFormat: string(crewsaml.PersistentNameIDFormat), Index: "idx-1",
		CustomAttributes: []crewsaml.Attribute{{Name: "email", Values: []crewsaml.AttributeValue{{Value: "ada@example.com"}}},
			{Name: "groups", Values: []crewsaml.AttributeValue{{Value: "eng"}, {Value: "ops"}}}}})
	return fixture{idp: idp, sp: sp}
}

// respond starts an SP-initiated sign-in at the IdP and returns its SAMLResponse and our request ID.
func (f fixture) respond(t *testing.T) (string, string) {
	t.Helper()
	target, reqID, err := f.sp.AuthnRedirect("relay-1")
	if err != nil {
		t.Fatal(err)
	}
	resp, err := f.idp.Server.Client().Get(target)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("idp returned %s: %s", resp.Status, body)
	}
	action, samlResponse, relay, err := samlfake.Form(body)
	if err != nil {
		t.Fatal(err)
	}
	if action != samlsp.ACSURL("https://starlight.example", "corp") || relay != "relay-1" {
		t.Fatalf("form posts to %q relay %q", action, relay)
	}
	return samlResponse, reqID
}

func TestValidResponseAccepted(t *testing.T) {
	f := newFixture(t, false)
	resp, reqID := f.respond(t)
	a, err := f.sp.Parse(resp, []string{reqID})
	if err != nil {
		t.Fatal(err)
	}
	if a.NameID != "persistent-ada" || a.SessionIndex != "idx-1" || a.Attributes["email"][0] != "ada@example.com" || len(a.Attributes["groups"]) != 2 {
		t.Fatalf("assertion %+v", a)
	}
	if a.ID == "" || a.NotOnOrAfter.Before(time.Now()) {
		t.Fatalf("assertion id %q expiry %v", a.ID, a.NotOnOrAfter)
	}
}

func TestRejectedResponses(t *testing.T) {
	cases := map[string]func(*crewsaml.Assertion){
		"wrong audience": func(a *crewsaml.Assertion) {
			a.Conditions.AudienceRestrictions[0].Audience.Value = "https://other.example"
		},
		"no audience": func(a *crewsaml.Assertion) { a.Conditions.AudienceRestrictions = nil },
		"expired": func(a *crewsaml.Assertion) {
			old := time.Now().Add(-2 * time.Hour)
			a.IssueInstant = old
			a.Conditions.NotBefore, a.Conditions.NotOnOrAfter = old, old.Add(time.Minute)
			a.Subject.SubjectConfirmations[0].SubjectConfirmationData.NotOnOrAfter = old.Add(time.Minute)
		},
		"wrong recipient": func(a *crewsaml.Assertion) {
			a.Subject.SubjectConfirmations[0].SubjectConfirmationData.Recipient = "https://evil.example/acs"
		},
		"no bearer": func(a *crewsaml.Assertion) {
			a.Subject.SubjectConfirmations[0].Method = "urn:oasis:names:tc:SAML:2.0:cm:holder-of-key"
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t, false)
			f.idp.Mutate(mutate)
			resp, reqID := f.respond(t)
			if _, err := f.sp.Parse(resp, []string{reqID}); err == nil {
				t.Fatal("accepted")
			}
		})
	}
}

// plain re-registers the SP without its encryption key so the IdP sends a readable, signed assertion.
func (f fixture) plain(t *testing.T) {
	t.Helper()
	md, err := f.sp.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	var ed crewsaml.EntityDescriptor
	if err := xml.Unmarshal(md, &ed); err != nil {
		t.Fatal(err)
	}
	for i := range ed.SPSSODescriptors {
		var keep []crewsaml.KeyDescriptor
		for _, k := range ed.SPSSODescriptors[i].KeyDescriptors {
			if k.Use != "encryption" {
				keep = append(keep, k)
			}
		}
		ed.SPSSODescriptors[i].KeyDescriptors = keep
	}
	b, _ := xml.Marshal(ed)
	if err := f.idp.RegisterSP(b); err != nil {
		t.Fatal(err)
	}
}

func TestEncryptedAssertionTamperRejected(t *testing.T) {
	f := newFixture(t, false)
	resp, reqID := f.respond(t)
	raw, _ := base64.StdEncoding.DecodeString(resp)
	if !strings.Contains(string(raw), "EncryptedAssertion") {
		t.Fatal("expected the IdP to encrypt to Starlight's published key")
	}
	i := strings.Index(string(raw), "<xenc:CipherValue>") + len("<xenc:CipherValue>") + 40
	flipped := []byte(string(raw))
	if flipped[i] == 'A' {
		flipped[i] = 'B'
	} else {
		flipped[i] = 'A'
	}
	if _, err := f.sp.Parse(base64.StdEncoding.EncodeToString(flipped), []string{reqID}); err == nil {
		t.Fatal("tampered ciphertext accepted")
	}
}

func TestTamperedAndMisdirectedResponses(t *testing.T) {
	f := newFixture(t, false)
	f.plain(t)
	resp, reqID := f.respond(t)
	raw, _ := base64.StdEncoding.DecodeString(resp)
	forged := strings.Replace(string(raw), "ada@example.com", "eve@example.com", 1)
	if forged == string(raw) {
		t.Fatal("fixture did not contain the email to tamper with")
	}
	if _, err := f.sp.Parse(base64.StdEncoding.EncodeToString([]byte(forged)), []string{reqID}); err == nil {
		t.Fatal("tampered response accepted")
	}
	if _, err := f.sp.Parse(resp, []string{"id-of-another-request"}); err == nil {
		t.Fatal("response for another request accepted")
	}
	if _, err := f.sp.Parse("not base64!", []string{reqID}); err == nil {
		t.Fatal("garbage accepted")
	}
}

func TestUnsolicitedResponses(t *testing.T) {
	for _, allow := range []bool{false, true} {
		f := newFixture(t, allow)
		target := f.idp.Server.URL + "/initiated?sp=" + samlsp.EntityID("https://starlight.example", "corp")
		resp, err := f.idp.Server.Client().Get(target)
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		_, samlResponse, _, err := samlfake.Form(body)
		if err != nil {
			t.Fatalf("idp-initiated form: %v %s", err, body)
		}
		_, err = f.sp.Parse(samlResponse, nil)
		if allow && err != nil {
			t.Fatalf("idp-initiated refused while allowed: %v", err)
		}
		if !allow && err == nil {
			t.Fatal("unsolicited response accepted while IdP-initiated sign-in is off")
		}
	}
}

func TestSHA1SignedResponsesRefused(t *testing.T) {
	for _, encrypted := range []bool{true, false} {
		f := newFixture(t, false)
		if !encrypted {
			f.plain(t)
		}
		f.idp.IDP.SignatureMethod = dsig.RSASHA1SignatureMethod
		resp, reqID := f.respond(t)
		if _, err := f.sp.Parse(resp, []string{reqID}); !errors.Is(err, samlsp.ErrInvalid) || !strings.Contains(err.Error(), "is not accepted") {
			t.Fatalf("RSA-SHA1 signed response, encrypted assertion %v: %v", encrypted, err)
		}
	}
}

const (
	nestSignature   = `<s:Signature><s:SignedInfo><s:CanonicalizationMethod Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/>`
	unnestSignature = `</s:SignedInfo><s:SignatureValue/></s:Signature>`
)

func TestResponseShapeCheckedBeforeSignatures(t *testing.T) {
	f := newFixture(t, true)
	head := `<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" xmlns:s="http://www.w3.org/2000/09/xmldsig#" ID="r"`
	var attrs strings.Builder
	for i := range 40 {
		fmt.Fprintf(&attrs, ` a%d=""`, i)
	}
	for name, tc := range map[string]struct{ xml, says string }{
		"nested signatures": {head + ">" + strings.Repeat(nestSignature, 50) + strings.Repeat(unnestSignature, 50) + "</samlp:Response>", "misplaced"},
		"deep":              {head + ">" + strings.Repeat("<a>", 40) + strings.Repeat("</a>", 40) + "</samlp:Response>", "nests too deep"},
		"wide":              {head + attrs.String() + "/>", "nests too deep"},
		"two assertions":    {head + ">" + strings.Repeat(`<a:EncryptedAssertion xmlns:a="urn:oasis:names:tc:SAML:2.0:assertion"/>`, 2) + "</samlp:Response>", "2 assertions"},
	} {
		if _, err := f.sp.Parse(base64.StdEncoding.EncodeToString([]byte(tc.xml)), nil); !errors.Is(err, samlsp.ErrInvalid) || !strings.Contains(err.Error(), tc.says) {
			t.Errorf("%s: %v", name, err)
		}
	}
	encrypted := encryptTo(t, f, `<saml:Assertion xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" xmlns:s="http://www.w3.org/2000/09/xmldsig#" ID="a">`+
		strings.Repeat(nestSignature, 10)+strings.Repeat(unnestSignature, 10)+`</saml:Assertion>`)
	response := `<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" ID="r" Version="2.0"` +
		` IssueInstant="` + time.Now().UTC().Format(time.RFC3339) + `" Destination="` + samlsp.ACSURL("https://starlight.example", "corp") + `">` +
		`<saml:Issuer>` + f.idp.IDP.MetadataURL.String() + `</saml:Issuer><samlp:Status><samlp:StatusCode Value="` + crewsaml.StatusSuccess + `"/></samlp:Status>` +
		encrypted + `</samlp:Response>`
	if _, err := f.sp.Parse(base64.StdEncoding.EncodeToString([]byte(response)), nil); !errors.Is(err, samlsp.ErrInvalid) || !strings.Contains(err.Error(), "misplaced") {
		t.Fatalf("nested signatures inside an encrypted assertion: %v", err)
	}
}

// encryptTo wraps plaintext in an EncryptedAssertion for Starlight's published encryption certificate, as anyone can.
func encryptTo(t *testing.T, f fixture, plaintext string) string {
	t.Helper()
	md, err := f.sp.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	var ed crewsaml.EntityDescriptor
	if err := xml.Unmarshal(md, &ed); err != nil {
		t.Fatal(err)
	}
	var cert *x509.Certificate
	for _, k := range ed.SPSSODescriptors[0].KeyDescriptors {
		if k.Use == "encryption" {
			der, _ := base64.StdEncoding.DecodeString(k.KeyInfo.X509Data.X509Certificates[0].Data)
			if cert, err = x509.ParseCertificate(der); err != nil {
				t.Fatal(err)
			}
		}
	}
	enc := xmlenc.OAEP()
	enc.BlockCipher = xmlenc.AES128CBC
	el, err := enc.Encrypt(cert, []byte(plaintext), nil)
	if err != nil {
		t.Fatal(err)
	}
	el.CreateAttr("Type", "http://www.w3.org/2001/04/xmlenc#Element")
	doc := etree.NewDocument()
	doc.CreateElement("saml:EncryptedAssertion").AddChild(el)
	out, err := doc.WriteToString()
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestParseMetadataAcceptsEntitiesDescriptor(t *testing.T) {
	idp := samlfake.New()
	defer idp.Close()
	one := idp.MetadataXML()
	wrapped := `<EntitiesDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata">` + strings.Replace(one, `<?xml version="1.0" encoding="UTF-8"?>`, "", 1) + `</EntitiesDescriptor>`
	for _, md := range []string{one, wrapped} {
		ed, err := samlsp.ParseMetadata([]byte(md))
		if err != nil || len(ed.IDPSSODescriptors) == 0 {
			t.Fatalf("metadata %v", err)
		}
	}
	if _, err := samlsp.ParseMetadata([]byte("<nope/>")); err == nil {
		t.Fatal("accepted metadata without an IdP")
	}
}
