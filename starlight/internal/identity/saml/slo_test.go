package saml_test

import (
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"encoding/pem"
	"encoding/xml"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/beevik/etree"
	crewsaml "github.com/crewjam/saml"
	dsig "github.com/russellhaering/goxmldsig"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
	"github.com/ashhart/TensorFold/starlight/internal/identity/saml/samlfake"
)

var sloURL = samlsp.SLOURL("https://starlight.example", "corp")

// sloFixture builds an SP for an IdP whose metadata lists logout endpoints for bindings only.
func sloFixture(t *testing.T, bindings ...string) fixture {
	t.Helper()
	idp := samlfake.New()
	t.Cleanup(idp.Close)
	idp.SetSLOBindings(bindings...)
	return build(t, idp, func(*connections.SAMLConfig) {})
}

func redirectBoth() []string { return []string{crewsaml.HTTPRedirectBinding, crewsaml.HTTPPostBinding} }

// message turns a delivery to Starlight's logout URL into the request Starlight's handler would see.
func message(t *testing.T, d samlfake.Delivery) samlsp.Message {
	t.Helper()
	var r *http.Request
	if d.Form == nil {
		r = httptest.NewRequest(http.MethodGet, d.URL, nil)
	} else {
		r = httptest.NewRequest(http.MethodPost, d.URL, strings.NewReader(d.Form.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	m, err := samlsp.ReadMessage(r)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// send carries Starlight's outbound message to the fake IdP and returns the IdP's reply, unfollowed.
func (f fixture) send(t *testing.T, out samlsp.Outbound) *http.Response {
	t.Helper()
	c := &http.Client{Transport: f.idp.Server.Client().Transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	var resp *http.Response
	var err error
	if out.Form == nil {
		resp, err = c.Get(out.URL)
	} else {
		resp, err = c.PostForm(out.URL, out.Form)
	}
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// reply runs an SP-initiated logout at the fake IdP and returns the IdP's answer as Starlight would receive it.
func (f fixture) reply(t *testing.T) (samlsp.Message, string) {
	t.Helper()
	out, id, err := f.sp.LogoutRequest(samlsp.NameID{Value: "persistent-ada", Format: string(crewsaml.PersistentNameIDFormat)}, "idx-1")
	if err != nil {
		t.Fatal(err)
	}
	resp := f.send(t, out)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	switch resp.StatusCode {
	case http.StatusFound:
		return message(t, samlfake.Delivery{URL: resp.Header.Get("Location")}), id
	case http.StatusOK:
		action, value, relay, err := samlfake.Form(body)
		if err != nil {
			t.Fatal(err)
		}
		form := url.Values{"SAMLResponse": {value}}
		if relay != "" {
			form.Set("RelayState", relay)
		}
		return message(t, samlfake.Delivery{URL: action, Form: form}), id
	}
	t.Fatalf("IdP refused Starlight's LogoutRequest: %d %s", resp.StatusCode, body)
	return samlsp.Message{}, ""
}

func TestSPMetadataAdvertisesSingleLogout(t *testing.T) {
	idp := samlfake.New()
	defer idp.Close()
	for _, sign := range []bool{true, false} {
		f := build(t, idp, func(s *connections.SAMLConfig) { s.SignRequests = sign })
		raw, err := f.sp.Metadata()
		if err != nil {
			t.Fatal(err)
		}
		var ed crewsaml.EntityDescriptor
		if err := xml.Unmarshal(raw, &ed); err != nil {
			t.Fatal(err)
		}
		d := ed.SPSSODescriptors[0]
		bindings := map[string]bool{}
		for _, e := range d.SingleLogoutServices {
			bindings[e.Binding] = e.Location == sloURL && e.ResponseLocation == sloURL
		}
		signing := false
		for _, k := range d.KeyDescriptors {
			signing = signing || k.Use == "signing"
		}
		if !bindings[crewsaml.HTTPRedirectBinding] || !bindings[crewsaml.HTTPPostBinding] || !signing {
			t.Fatalf("sign requests %v: logout endpoints %v, signing key %v", sign, bindings, signing)
		}
	}
}

func TestSPInitiatedLogoutOverEachBinding(t *testing.T) {
	cases := map[string]struct {
		bindings []string
		reply    samlfake.Logout
	}{
		"redirect":                 {bindings: redirectBoth()},
		"post request":             {bindings: []string{crewsaml.HTTPPostBinding}},
		"redirect with post reply": {bindings: redirectBoth(), reply: samlfake.Logout{POST: true}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			f := sloFixture(t, tc.bindings...)
			f.idp.SetLogoutReply(tc.reply)
			m, id := f.reply(t)
			resp, err := f.sp.VerifyLogoutResponse(m)
			if err != nil || resp.InResponseTo != id || !resp.Success() {
				t.Fatalf("response %+v, %v (want InResponseTo %s)", resp, err, id)
			}
			seen := f.idp.LogoutRequests()
			if len(seen) != 1 || seen[0].Err != nil || seen[0].ID != id || seen[0].NameID.Value != "persistent-ada" ||
				seen[0].NameID.Format != string(crewsaml.PersistentNameIDFormat) || seen[0].SessionIndex != "idx-1" {
				t.Fatalf("IdP saw %+v", seen)
			}
		})
	}
}

func TestLogoutResponseChecks(t *testing.T) {
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	cases := map[string]samlfake.Logout{
		"wrong key":         {Key: other},
		"wrong key post":    {Key: other, POST: true},
		"unsigned redirect": {Unsigned: true},
		"unsigned post":     {Unsigned: true, POST: true},
		"wrong issuer":      {Issuer: "https://idp.example.org/other"},
		"wrong destination": {Destination: "https://starlight.example/auth/other/saml/slo"},
		"stale":             {IssueInstant: time.Now().Add(-time.Hour)},
		"from the future":   {IssueInstant: time.Now().Add(time.Hour)},
	}
	for name, l := range cases {
		t.Run(name, func(t *testing.T) {
			f := sloFixture(t, redirectBoth()...)
			f.idp.SetLogoutReply(l)
			m, _ := f.reply(t)
			if _, err := f.sp.VerifyLogoutResponse(m); !errors.Is(err, samlsp.ErrLogout) {
				t.Fatalf("accepted or wrong error: %v", err)
			}
		})
	}
	f := sloFixture(t, redirectBoth()...)
	f.idp.SetLogoutReply(samlfake.Logout{Status: crewsaml.StatusRequester})
	m, _ := f.reply(t)
	if resp, err := f.sp.VerifyLogoutResponse(m); err != nil || resp.Success() {
		t.Fatalf("a signed failure status must verify but not count as success: %+v %v", resp, err)
	}
	if _, err := f.sp.VerifyLogoutRequest(m); err == nil {
		t.Fatal("a SAMLResponse verified as a LogoutRequest")
	}
}

func TestIDPInitiatedLogoutRequest(t *testing.T) {
	for _, post := range []bool{false, true} {
		f := sloFixture(t, redirectBoth()...)
		d, err := f.idp.LogoutRequest(samlfake.Logout{SessionIndexes: []string{"idx-1", "idx-2"}, RelayState: "relay-9", POST: post})
		if err != nil {
			t.Fatal(err)
		}
		m := message(t, d)
		req, err := f.sp.VerifyLogoutRequest(m)
		if err != nil {
			t.Fatalf("post %v: %v", post, err)
		}
		if req.NameID != "persistent-ada" || len(req.SessionIndexes) != 2 || req.SessionIndexes[1] != "idx-2" || req.ID == "" ||
			time.Until(req.Expires) < 5*time.Minute || m.RelayState != "relay-9" {
			t.Fatalf("request %+v relay %q", req, m.RelayState)
		}
		out, err := f.sp.LogoutResponse(req.ID, m.RelayState)
		if err != nil {
			t.Fatal(err)
		}
		f.send(t, out).Body.Close()
		got := f.idp.LogoutResponses()
		if len(got) != 1 || got[0].Err != nil || got[0].InResponseTo != req.ID || got[0].Status != crewsaml.StatusSuccess ||
			got[0].RelayState != "relay-9" || got[0].Binding != "redirect" {
			t.Fatalf("IdP saw %+v", got)
		}
	}
	f := sloFixture(t, crewsaml.HTTPPostBinding)
	out, err := f.sp.LogoutResponse("_req-1", "")
	if err != nil || out.Form == nil {
		t.Fatalf("post-only IdP: %+v %v", out, err)
	}
	f.send(t, out).Body.Close()
	if got := f.idp.LogoutResponses(); len(got) != 1 || got[0].Err != nil || got[0].Binding != "post" {
		t.Fatalf("IdP saw %+v", got)
	}
}

func TestForgedLogoutRequestsRefused(t *testing.T) {
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	cases := map[string]samlfake.Logout{
		"wrong key":         {Key: other},
		"wrong key post":    {Key: other, POST: true},
		"unsigned redirect": {Unsigned: true},
		"unsigned post":     {Unsigned: true, POST: true},
		"wrong issuer":      {Issuer: "https://idp.example.org/other"},
		"wrong destination": {Destination: "https://starlight.example/auth/other/saml/slo"},
		"stale":             {IssueInstant: time.Now().Add(-time.Hour)},
	}
	for name, l := range cases {
		t.Run(name, func(t *testing.T) {
			f := sloFixture(t, redirectBoth()...)
			d, err := f.idp.LogoutRequest(l)
			if err != nil {
				t.Fatal(err)
			}
			if d.Form == nil && l.Destination != "" {
				d.URL = sloURL + d.URL[strings.Index(d.URL, "?"):]
			}
			if _, err := f.sp.VerifyLogoutRequest(message(t, d)); !errors.Is(err, samlsp.ErrLogout) {
				t.Fatalf("accepted or wrong error: %v", err)
			}
		})
	}
	f := sloFixture(t, redirectBoth()...)
	d, _ := f.idp.LogoutRequest(samlfake.Logout{NameID: "persistent-bo", POST: true})
	raw, _ := base64.StdEncoding.DecodeString(d.Form.Get("SAMLRequest"))
	d.Form.Set("SAMLRequest", base64.StdEncoding.EncodeToString([]byte(strings.Replace(string(raw), "persistent-bo", "persistent-ada", 1))))
	if _, err := f.sp.VerifyLogoutRequest(message(t, d)); err == nil {
		t.Fatal("tampered NameID accepted")
	}
	d, _ = f.idp.LogoutRequest(samlfake.Logout{RelayState: "a"})
	d.URL = strings.Replace(d.URL, "RelayState=a", "RelayState=b", 1)
	if _, err := f.sp.VerifyLogoutRequest(message(t, d)); err == nil {
		t.Fatal("tampered RelayState accepted")
	}
}

func TestNoSingleLogoutEndpoint(t *testing.T) {
	f := sloFixture(t)
	if f.sp.HasSLO() {
		t.Fatal("HasSLO with no logout endpoint in the metadata")
	}
	if _, _, err := f.sp.LogoutRequest(samlsp.NameID{Value: "persistent-ada"}, ""); !errors.Is(err, samlsp.ErrNoSLO) {
		t.Fatalf("logout request = %v", err)
	}
	if _, err := f.sp.LogoutResponse("_r", ""); !errors.Is(err, samlsp.ErrNoSLO) {
		t.Fatalf("logout response = %v", err)
	}
}

func TestReadMessageRejectsAmbiguousQueries(t *testing.T) {
	for _, q := range []string{"", "RelayState=x", "SAMLRequest=a&SAMLResponse=b", "SAMLRequest=a&SAMLRequest=b", "SAMLRequest=%zz"} {
		if _, err := samlsp.ReadMessage(httptest.NewRequest(http.MethodGet, sloURL+"?"+q, nil)); err == nil {
			t.Errorf("query %q accepted", q)
		}
	}
}

// signedPOST is a signed HTTP-POST LogoutRequest from the fake IdP, as XML, with the delivery that carries it.
func signedPOST(t *testing.T, f fixture, l samlfake.Logout) (string, samlfake.Delivery) {
	t.Helper()
	l.POST = true
	d, err := f.idp.LogoutRequest(l)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := base64.StdEncoding.DecodeString(d.Form.Get("SAMLRequest"))
	return string(raw), d
}

func TestLogoutSignatureMustSitAloneUnderTheRoot(t *testing.T) {
	f := sloFixture(t, redirectBoth()...)
	signed, d := signedPOST(t, f, samlfake.Logout{})
	sig := signed[strings.Index(signed, "<ds:Signature") : strings.Index(signed, "</ds:Signature>")+len("</ds:Signature>")]
	nest := `<ds:Signature><ds:SignedInfo><ds:CanonicalizationMethod Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/>`
	unnest := `</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue></ds:Signature>`
	for name, tc := range map[string]struct{ xml, says string }{
		"nested":      {strings.Replace(signed, "<ds:SignedInfo>", "<ds:SignedInfo>"+strings.Repeat(nest, 20)+strings.Repeat(unnest, 20), 1), "misplaced"},
		"twice":       {strings.Replace(signed, sig, sig+sig, 1), "misplaced"},
		"tucked away": {strings.Replace(signed, sig, "<samlp:Extensions>"+sig+"</samlp:Extensions>", 1), "misplaced"},
		"deep":        {strings.Replace(signed, "</samlp:LogoutRequest>", strings.Repeat("<x>", 40)+strings.Repeat("</x>", 40)+"</samlp:LogoutRequest>", 1), "nests too deep"},
	} {
		d.Form.Set("SAMLRequest", base64.StdEncoding.EncodeToString([]byte(tc.xml)))
		if _, err := f.sp.VerifyLogoutRequest(message(t, d)); !errors.Is(err, samlsp.ErrLogout) || !strings.Contains(err.Error(), tc.says) {
			t.Errorf("%s: %v, want a refusal before any signature check", name, err)
		}
	}
}

func TestLogoutMessagesLimitedTo32KiB(t *testing.T) {
	f := sloFixture(t, redirectBoth()...)
	for _, post := range []bool{true, false} {
		d, err := f.idp.LogoutRequest(samlfake.Logout{SessionIndexes: []string{strings.Repeat("i", 40<<10)}, POST: post})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.sp.VerifyLogoutRequest(message(t, d)); !errors.Is(err, samlsp.ErrLogout) || !strings.Contains(err.Error(), "too large") {
			t.Errorf("a 40 KiB signed logout message over post %v = %v", post, err)
		}
	}
	_, d := signedPOST(t, f, samlfake.Logout{SessionIndexes: []string{strings.Repeat("i", 16<<10)}})
	if _, err := f.sp.VerifyLogoutRequest(message(t, d)); err != nil {
		t.Fatalf("a signed logout message under the limit: %v", err)
	}
}

func TestPOSTLogoutNeedsSHA256OrStronger(t *testing.T) {
	f := sloFixture(t, redirectBoth()...)
	el := etree.NewElement("samlp:LogoutRequest")
	el.CreateAttr("xmlns:samlp", "urn:oasis:names:tc:SAML:2.0:protocol")
	el.CreateAttr("xmlns:saml", "urn:oasis:names:tc:SAML:2.0:assertion")
	el.CreateAttr("ID", "_sha1req")
	el.CreateAttr("Version", "2.0")
	el.CreateAttr("IssueInstant", time.Now().UTC().Format(time.RFC3339))
	el.CreateAttr("Destination", sloURL)
	el.CreateElement("saml:Issuer").SetText(f.idp.IDP.MetadataURL.String())
	el.CreateElement("saml:NameID").SetText("persistent-ada")
	ctx, err := dsig.NewSigningContext(f.idp.IDP.Key.(*rsa.PrivateKey), [][]byte{f.idp.IDP.Certificate.Raw})
	if err != nil {
		t.Fatal(err)
	}
	ctx.Canonicalizer = dsig.MakeC14N10ExclusiveCanonicalizerWithPrefixList("")
	if err := ctx.SetSignatureMethod(dsig.RSASHA1SignatureMethod); err != nil {
		t.Fatal(err)
	}
	signed, err := ctx.SignEnveloped(el)
	if err != nil {
		t.Fatal(err)
	}
	doc := etree.NewDocument()
	doc.SetRoot(signed)
	sha1, _ := doc.WriteToString()
	sha256Signed, d := signedPOST(t, f, samlfake.Logout{})
	weakDigest := strings.Replace(sha256Signed, "http://www.w3.org/2001/04/xmlenc#sha256", "http://www.w3.org/2000/09/xmldsig#sha1", 1)
	for name, xml := range map[string]string{"RSA-SHA1 over a SHA-1 digest": sha1, "SHA-1 digest": weakDigest} {
		d.Form.Set("SAMLRequest", base64.StdEncoding.EncodeToString([]byte(xml)))
		if _, err := f.sp.VerifyLogoutRequest(message(t, d)); !errors.Is(err, samlsp.ErrLogout) || !strings.Contains(err.Error(), "is not accepted") {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// withSigningCert lists another current signing certificate ahead of the IdP's own in its metadata.
func withSigningCert(t *testing.T, metadata string) string {
	t.Helper()
	certPEM, _, err := connections.NewSPKeyPair("another-idp-key")
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode([]byte(certPEM))
	var ed crewsaml.EntityDescriptor
	if err := xml.Unmarshal([]byte(metadata), &ed); err != nil {
		t.Fatal(err)
	}
	other := crewsaml.KeyDescriptor{Use: "signing", KeyInfo: crewsaml.KeyInfo{X509Data: crewsaml.X509Data{
		X509Certificates: []crewsaml.X509Certificate{{Data: base64.StdEncoding.EncodeToString(block.Bytes)}}}}}
	d := &ed.IDPSSODescriptors[0]
	d.KeyDescriptors = append([]crewsaml.KeyDescriptor{other}, d.KeyDescriptors...)
	b, err := xml.Marshal(ed)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestLogoutVerifiesAgainstEachIdPCertificate(t *testing.T) {
	idp := samlfake.New()
	t.Cleanup(idp.Close)
	f := build(t, idp, func(s *connections.SAMLConfig) { s.MetadataXML = withSigningCert(t, s.MetadataXML) })
	_, d := signedPOST(t, f, samlfake.Logout{})
	if _, err := f.sp.VerifyLogoutRequest(message(t, d)); err != nil {
		t.Fatalf("signed by the second of two certificates, which it carries: %v", err)
	}
	signed, d := signedPOST(t, f, samlfake.Logout{})
	bare := signed[:strings.Index(signed, "<ds:KeyInfo>")] + signed[strings.Index(signed, "</ds:KeyInfo>")+len("</ds:KeyInfo>"):]
	d.Form.Set("SAMLRequest", base64.StdEncoding.EncodeToString([]byte(bare)))
	if _, err := f.sp.VerifyLogoutRequest(message(t, d)); err != nil {
		t.Fatalf("signed by the second of two certificates, without KeyInfo: %v", err)
	}
}
