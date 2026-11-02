package login

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	crewsaml "github.com/crewjam/saml"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
	"github.com/ashhart/TensorFold/starlight/internal/identity/saml/samlfake"
)

func (x *harness) samlConn(idp *samlfake.IdP, allowInitiated bool) connections.Connection {
	x.t.Helper()
	c, err := x.h.Connections().Create(context.Background(), connections.Input{Slug: "corp-saml", Name: "Corp SAML", Protocol: connections.SAML,
		Preset: "generic_saml", Enabled: true, JIT: true,
		Config: connections.Config{SAML: &connections.SAMLConfig{MetadataXML: idp.MetadataXML(), SignRequests: true, NameIDFormat: "persistent",
			AllowIDPInitiated: allowInitiated}, Claims: connections.ClaimMap{Email: "email", Name: "name", Groups: "groups"},
			Policy: connections.Policy{TrustEmail: true}}}, nil)
	if err != nil {
		x.t.Fatal(err)
	}
	code, md := x.get(x.client, "/auth/corp-saml/saml/metadata")
	if code != 200 || !strings.Contains(md, samlsp.ACSURL(x.starlight.URL, "corp-saml")) {
		x.t.Fatalf("SP metadata = %d", code)
	}
	if err := idp.RegisterSP([]byte(md)); err != nil {
		x.t.Fatal(err)
	}
	idp.SetSession(crewsaml.Session{NameID: "persist-ada", NameIDFormat: string(crewsaml.PersistentNameIDFormat), Index: "idx-1",
		CustomAttributes: []crewsaml.Attribute{
			{Name: "email", Values: []crewsaml.AttributeValue{{Value: "ada@example.com"}}},
			{Name: "groups", Values: []crewsaml.AttributeValue{{Value: "eng"}}}}})
	return c
}

// postForm submits an IdP auto-post form the way a browser would.
func (x *harness) postForm(c *http.Client, body []byte) (int, string) {
	x.t.Helper()
	action, response, relay, err := samlfake.Form(body)
	if err != nil {
		x.t.Fatalf("%v: %s", err, body)
	}
	resp, err := c.PostForm(action, url.Values{"SAMLResponse": {response}, "RelayState": {relay}})
	if err != nil {
		x.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func TestSAMLSignInEndToEnd(t *testing.T) {
	x := newHarness(t, 100)
	idp := samlfake.New()
	defer idp.Close()
	x.samlConn(idp, false)
	resp, err := x.client.Get(x.starlight.URL + "/auth/corp-saml/start")
	if err != nil {
		t.Fatal(err)
	}
	form, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if code, body := x.postForm(x.client, form); code != 200 || body != "hello ada@example.com" {
		t.Fatalf("saml sign-in = %d %q", code, body)
	}
	if g := x.groups(x.user("ada@example.com").ID); !g["eng"] {
		t.Fatalf("saml groups %v", g)
	}
	if code, _ := x.postForm(x.client, form); code == 200 {
		t.Fatal("replayed SP-initiated response accepted")
	}
	if code, _ := x.postForm(x.browser(), form); code == 200 {
		t.Fatal("response accepted in a browser without the binding cookie")
	}
}

func TestSAMLUnsolicitedAndReplayedAssertions(t *testing.T) {
	x := newHarness(t, 100)
	idp := samlfake.New()
	defer idp.Close()
	c := x.samlConn(idp, false)
	initiated := func() []byte {
		resp, err := idp.Server.Client().Get(idp.Server.URL + "/initiated?sp=" + url.QueryEscape(samlsp.EntityID(x.starlight.URL, "corp-saml")))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return b
	}
	if code, _ := x.postForm(x.browser(), initiated()); code == 200 {
		t.Fatal("unsolicited response accepted while IdP-initiated sign-in is off")
	}
	in := connections.Input{Slug: c.Slug, Name: c.Name, Protocol: c.Protocol, Preset: c.Preset, Enabled: true, JIT: true, Config: c.Config}
	in.Config.SAML.AllowIDPInitiated = true
	if _, err := x.h.Connections().Update(context.Background(), c.ID, in, nil); err != nil {
		t.Fatal(err)
	}
	form := initiated()
	if code, body := x.postForm(x.browser(), form); code != 200 || !strings.Contains(body, "hello") {
		t.Fatalf("allowed IdP-initiated = %d %s", code, body)
	}
	if code, _ := x.postForm(x.browser(), form); code == 200 {
		t.Fatal("replayed assertion accepted")
	}
}

func TestSAMLMutatedAssertionsRefused(t *testing.T) {
	x := newHarness(t, 100)
	idp := samlfake.New()
	defer idp.Close()
	x.samlConn(idp, false)
	idp.Mutate(func(a *crewsaml.Assertion) {
		a.Conditions.AudienceRestrictions[0].Audience.Value = "https://other.example/sp"
	})
	resp, err := x.client.Get(x.starlight.URL + "/auth/corp-saml/start")
	if err != nil {
		t.Fatal(err)
	}
	form, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if code, body := x.postForm(x.client, form); code != 401 || !strings.Contains(body, "could not verify") {
		t.Fatalf("wrong audience = %d", code)
	}
}

func TestSAMLPostsLimitedPerClientNetwork(t *testing.T) {
	x := newHarness(t, 2)
	idp := samlfake.New()
	defer idp.Close()
	x.samlConn(idp, true)
	junk := url.Values{"SAMLRequest": {"bm90IGEgbG9nb3V0IG1lc3NhZ2U="}}
	for i := range 4 {
		if code, _ := x.post(x.browser(), "/auth/corp-saml/saml/slo", junk); code != http.StatusBadRequest {
			t.Fatalf("logout post %d = %d", i+1, code)
		}
	}
	if code, _ := x.post(x.browser(), "/auth/corp-saml/saml/slo", junk); code != http.StatusTooManyRequests {
		t.Fatalf("logout post past the budget = %d", code)
	}
	if code, _ := x.post(x.browser(), "/auth/corp-saml/saml/acs", url.Values{"SAMLResponse": {"bm90IGEgcmVzcG9uc2U="}}); code != http.StatusTooManyRequests {
		t.Fatalf("ACS post past the shared budget = %d", code)
	}
	if code, _ := x.get(x.browser(), "/auth/corp-saml/start"); code == http.StatusTooManyRequests {
		t.Fatal("logout and ACS posts spent the sign-in start budget")
	}
}
