package console

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestFrameworkGuardsAndRenders(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "ivy@example.com", Source: "jit"})
	if _, err := pool.Exec(ctx, "INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, 'auditor', 'user', $2)", ids.New(), u.ID); err != nil {
		t.Fatal(err)
	}
	sessions := session.New(pool, session.Config{})
	proxies, _ := httpx.NewProxies(nil)
	c, err := New(&Console{Pool: pool, Sessions: sessions, Proxies: proxies, Org: func(context.Context) string { return "Example Corp" }, Version: "test"})
	if err != nil {
		t.Fatal(err)
	}
	c.Handle("GET /console/probe", authz.AuditRead, func(q *Request) error {
		q.Render("message", "Probe", messagePage{Heading: "Probe <ok>", Body: "visible to auditors"})
		return nil
	})
	c.Handle("POST /console/probe", authz.AuditRead, func(q *Request) error { return q.Redirect("/console/probe", "Saved") })
	c.Handle("GET /console/secret", authz.SSOWrite, func(q *Request) error { return nil })
	mux := http.NewServeMux()
	c.Register(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()
	rec := httptest.NewRecorder()
	s, err := sessions.Create(ctx, rec, httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID})
	if err != nil {
		t.Fatal(err)
	}
	cookie := rec.Result().Cookies()[0]
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	do := func(method, path string, form url.Values) (*http.Response, string) {
		body := strings.NewReader("")
		if form != nil {
			body = strings.NewReader(form.Encode())
		}
		req, _ := http.NewRequest(method, srv.URL+path, body)
		req.AddCookie(cookie)
		if form != nil {
			req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			req.Header.Set("Origin", srv.URL)
		}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp, string(b)
	}
	resp, body := do("GET", "/console/probe", nil)
	hidden := strings.Contains(body, "/console/access") || strings.Contains(body, "/console/limits")
	if resp.StatusCode != 200 || !strings.Contains(body, "Probe &lt;ok&gt;") || !strings.Contains(body, `href="/console/audit"`) || hidden {
		t.Fatalf("render %d %s", resp.StatusCode, body)
	}
	if resp, _ := do("GET", "/console/secret", nil); resp.StatusCode != http.StatusForbidden {
		t.Fatalf("permission check answered %d", resp.StatusCode)
	}
	if resp, _ := do("POST", "/console/probe", url.Values{"_csrf": {"forged"}}); resp.StatusCode != http.StatusForbidden {
		t.Fatalf("forged CSRF answered %d", resp.StatusCode)
	}
	resp, _ = do("POST", "/console/probe", url.Values{"_csrf": {s.CSRF()}})
	if resp.StatusCode != http.StatusSeeOther || len(resp.Cookies()) == 0 {
		t.Fatalf("post %d", resp.StatusCode)
	}
}
