package login

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	crewsaml "github.com/crewjam/saml"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/saml/samlfake"
)

// samlLogoutConn is the harness SAML connection with "Sign out of the IdP too" switched on.
func (x *harness) samlLogoutConn(idp *samlfake.IdP) connections.Connection {
	x.t.Helper()
	c := x.samlConn(idp, false)
	in := connections.Input{Slug: c.Slug, Name: c.Name, Protocol: c.Protocol, Preset: c.Preset, Enabled: true, JIT: true, Config: c.Config}
	in.Config.SAML.RPLogout = true
	c, err := x.h.Connections().Update(context.Background(), c.ID, in, nil)
	if err != nil {
		x.t.Fatal(err)
	}
	return c
}

// samlSignIn runs an SP-initiated SAML sign-in in the browser as whoever the IdP's session names.
func (x *harness) samlSignIn(c *http.Client) {
	x.t.Helper()
	resp, err := c.Get(x.starlight.URL + "/auth/corp-saml/start")
	if err != nil {
		x.t.Fatal(err)
	}
	form, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if code, body := x.postForm(c, form); code != 200 || !strings.HasPrefix(body, "hello ") {
		x.t.Fatalf("saml sign-in = %d %q", code, body)
	}
}

// samlLogout signs the browser out of Starlight and returns the last page and the IdP's reply URL if the IdP took part.
func (x *harness) samlLogout(c *http.Client) (int, string, string) {
	x.t.Helper()
	var reply string
	watch := *c
	watch.CheckRedirect = func(req *http.Request, _ []*http.Request) error {
		if strings.HasSuffix(req.URL.Path, "/saml/slo") {
			reply = req.URL.String()
		}
		return nil
	}
	code, body := x.post(&watch, "/auth/logout", url.Values{"_csrf": {x.session(c).CSRF()}})
	return code, body, reply
}

// deliver carries an IdP-initiated message to Starlight in a browser, following redirects.
func (x *harness) deliver(c *http.Client, d samlfake.Delivery) (int, string) {
	x.t.Helper()
	var resp *http.Response
	var err error
	if d.Form == nil {
		resp, err = c.Get(d.URL)
	} else {
		resp, err = c.PostForm(d.URL, d.Form)
	}
	if err != nil {
		x.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func (x *harness) count(sql string, args ...any) int {
	x.t.Helper()
	var n int
	if err := x.pool.QueryRow(context.Background(), sql, args...).Scan(&n); err != nil {
		x.t.Fatal(err)
	}
	return n
}

func TestSAMLSingleLogoutFromStarlight(t *testing.T) {
	x := newHarness(t, 100)
	idp := samlfake.New()
	defer idp.Close()
	x.samlLogoutConn(idp)
	x.samlSignIn(x.client)
	code, body, reply := x.samlLogout(x.client)
	if code != 200 || !strings.Contains(body, "You have signed out.") || reply == "" {
		t.Fatalf("logout = %d reply %q: %.300s", code, reply, body)
	}
	seen := idp.LogoutRequests()
	if len(seen) != 1 || seen[0].Err != nil || seen[0].NameID.Value != "persist-ada" ||
		seen[0].NameID.Format != string(crewsaml.PersistentNameIDFormat) || seen[0].SessionIndex != "idx-1" || seen[0].Binding != "redirect" {
		t.Fatalf("IdP saw %+v", seen)
	}
	if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
		t.Fatalf("session survived sign-out: %d", code)
	}
	if x.audits("auth.logout", "success") != 1 || x.audits("auth.logout_response", "success") != 1 {
		t.Fatal("logout and the IdP's confirmation are not both audited")
	}
	if n := x.count("SELECT count(*) FROM saml_logout_requests"); n != 0 {
		t.Fatalf("%d logout requests still pending after the answer", n)
	}
	if code, body := x.get(x.browser(), strings.TrimPrefix(reply, x.starlight.URL)); code != http.StatusBadRequest || !strings.Contains(body, "not confirmed") {
		t.Fatalf("replayed LogoutResponse = %d", code)
	}
	if x.audits("auth.logout_response", "failure") != 1 {
		t.Fatal("refused LogoutResponse not audited")
	}
}

func TestSAMLLogoutStaysLocalWithoutSLO(t *testing.T) {
	for _, tc := range []struct {
		name     string
		bindings []string
		optIn    bool
	}{
		{"opted out", []string{crewsaml.HTTPRedirectBinding}, false},
		{"no SLO endpoint", nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := newHarness(t, 100)
			idp := samlfake.New()
			defer idp.Close()
			idp.SetSLOBindings(tc.bindings...)
			if tc.optIn {
				x.samlLogoutConn(idp)
			} else {
				x.samlConn(idp, false)
			}
			x.samlSignIn(x.client)
			code, body, reply := x.samlLogout(x.client)
			if code != 200 || !strings.Contains(body, "You have signed out.") || reply != "" || len(idp.LogoutRequests()) != 0 {
				t.Fatalf("logout = %d reply %q, IdP saw %d requests", code, reply, len(idp.LogoutRequests()))
			}
			if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
				t.Fatal("session survived sign-out")
			}
			if x.audits("auth.logout", "success") != 1 || x.count("SELECT count(*) FROM saml_logout_requests") != 0 {
				t.Fatal("local sign-out not audited, or a logout request left pending")
			}
		})
	}
}

func TestSAMLForgedLogoutResponsesRefused(t *testing.T) {
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	cases := map[string]samlfake.Logout{
		"wrong signature":  {Key: other},
		"unknown request":  {InResponseTo: "id-never-sent"},
		"stale":            {IssueInstant: time.Now().Add(-time.Hour)},
		"post wrong issue": {POST: true, Issuer: "https://idp.example.org/other"},
	}
	for name, l := range cases {
		t.Run(name, func(t *testing.T) {
			x := newHarness(t, 100)
			idp := samlfake.New()
			defer idp.Close()
			x.samlLogoutConn(idp)
			idp.SetLogoutReply(l)
			x.samlSignIn(x.client)
			code, body, _ := x.samlLogout(x.client)
			if l.POST {
				code, body = x.postForm(x.client, []byte(body))
			}
			if code != http.StatusBadRequest || !strings.Contains(body, "not confirmed") {
				t.Fatalf("forged LogoutResponse = %d: %.200s", code, body)
			}
			if x.audits("auth.logout_response", "failure") != 1 || x.audits("auth.logout_response", "success") != 0 {
				t.Fatal("forged LogoutResponse not audited as a failure")
			}
			if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
				t.Fatal("the Starlight session must end even when the IdP's answer is refused")
			}
		})
	}
}

func TestSAMLLogoutFromTheIdP(t *testing.T) {
	x := newHarness(t, 100)
	idp := samlfake.New()
	defer idp.Close()
	x.samlConn(idp, false)
	ada1, ada2, bo := x.client, x.browser(), x.browser()
	x.samlSignIn(ada1)
	idp.SetSession(crewsaml.Session{NameID: "persist-ada", NameIDFormat: string(crewsaml.PersistentNameIDFormat), Index: "idx-2",
		CustomAttributes: []crewsaml.Attribute{{Name: "email", Values: []crewsaml.AttributeValue{{Value: "ada@example.com"}}}}})
	x.samlSignIn(ada2)
	idp.SetSession(crewsaml.Session{NameID: "persist-bo", NameIDFormat: string(crewsaml.PersistentNameIDFormat), Index: "idx-9",
		CustomAttributes: []crewsaml.Attribute{{Name: "email", Values: []crewsaml.AttributeValue{{Value: "bo@example.com"}}}}})
	x.samlSignIn(bo)
	live := func(c *http.Client) bool {
		code, _ := x.get(c, "/console")
		return code == http.StatusOK
	}
	byIndex, err := idp.LogoutRequest(samlfake.Logout{NameID: "persist-ada", SessionIndexes: []string{"idx-1"}, RelayState: "back-to-portal"})
	if err != nil {
		t.Fatal(err)
	}
	if code, body := x.deliver(x.browser(), byIndex); code != 200 || !strings.Contains(body, "signed out at the IdP") {
		t.Fatalf("IdP-initiated logout = %d %.200s", code, body)
	}
	if live(ada1) || !live(ada2) || !live(bo) {
		t.Fatalf("SessionIndex logout ended the wrong sessions: %v %v %v", live(ada1), live(ada2), live(bo))
	}
	answers := idp.LogoutResponses()
	if len(answers) != 1 || answers[0].Err != nil || answers[0].Status != crewsaml.StatusSuccess || answers[0].RelayState != "back-to-portal" {
		t.Fatalf("IdP received %+v", answers)
	}
	if code, _ := x.deliver(x.browser(), byIndex); code != http.StatusBadRequest {
		t.Fatalf("replayed LogoutRequest = %d", code)
	}
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	for name, l := range map[string]samlfake.Logout{
		"forged": {NameID: "persist-bo", Key: other, POST: true},
		"stale":  {NameID: "persist-bo", IssueInstant: time.Now().Add(-time.Hour)},
	} {
		d, err := idp.LogoutRequest(l)
		if err != nil {
			t.Fatal(err)
		}
		if code, _ := x.deliver(x.browser(), d); code != http.StatusBadRequest || !live(bo) {
			t.Fatalf("%s LogoutRequest = %d, bo live %v", name, code, live(bo))
		}
	}
	all, err := idp.LogoutRequest(samlfake.Logout{NameID: "persist-ada", POST: true})
	if err != nil {
		t.Fatal(err)
	}
	if code, _ := x.deliver(x.browser(), all); code != 200 || live(ada2) || !live(bo) {
		t.Fatalf("NameID logout = %d: ada %v bo %v", code, live(ada2), live(bo))
	}
	if x.audits("auth.idp_logout", "success") != 2 || x.audits("auth.idp_logout", "failure") != 3 {
		t.Fatalf("idp logouts audited %d ok, %d refused", x.audits("auth.idp_logout", "success"), x.audits("auth.idp_logout", "failure"))
	}
}

func TestRemovingAConnectionSignsItsUsersOut(t *testing.T) {
	x := newHarness(t, 100)
	idp := samlfake.New()
	defer idp.Close()
	c := x.samlConn(idp, false)
	ctx := context.Background()
	x.samlSignIn(x.client)
	if err := x.h.Connections().SetEnabled(ctx, c.ID, false, nil); err != nil {
		t.Fatal(err)
	}
	if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
		t.Fatalf("session survived disabling its connection: %d", code)
	}
	if err := x.h.Connections().SetEnabled(ctx, c.ID, true, nil); err != nil {
		t.Fatal(err)
	}
	x.samlSignIn(x.client)
	if err := x.h.Connections().Delete(ctx, c.ID, nil); err != nil {
		t.Fatal(err)
	}
	x.h.Forget(c.ID)
	if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
		t.Fatalf("session survived deleting its connection: %d", code)
	}
	if n := x.count("SELECT count(*) FROM sessions WHERE revoked_at IS NULL"); n != 0 {
		t.Fatalf("%d sessions still live", n)
	}
}
