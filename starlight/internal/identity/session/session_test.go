package session_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestLifecycleAndRevocation(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "ada@example.com", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	m := session.New(pool, session.Config{Idle: time.Hour, Absolute: 2 * time.Hour})
	w := httptest.NewRecorder()
	s, err := m.Create(ctx, w, httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID, AMR: []string{"mfa"}})
	if err != nil {
		t.Fatal(err)
	}
	cookie := w.Result().Cookies()[0]
	if !cookie.HttpOnly || cookie.SameSite != http.SameSiteLaxMode {
		t.Fatalf("cookie flags %+v", cookie)
	}
	r := httptest.NewRequest("GET", "/console", nil)
	r.AddCookie(cookie)
	got, user, err := m.Load(ctx, r)
	if err != nil || got.ID != s.ID || user.Email != "ada@example.com" || !got.ValidCSRF(s.CSRF()) {
		t.Fatalf("load = %+v %+v %v", got, user, err)
	}
	if got.ValidCSRF("forged") {
		t.Fatal("accepted a forged CSRF token")
	}
	if err := directory.RevokeUser(ctx, pool, u.ID, "test"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.Load(ctx, r); err != session.ErrNoSession {
		t.Fatalf("revoked session still loads: %v", err)
	}
}

func TestRequireEnforcesCSRFAndOrigin(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "bo@example.com", Source: "jit"})
	m := session.New(pool, session.Config{})
	w := httptest.NewRecorder()
	s, _ := m.Create(ctx, w, httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID})
	cookie := w.Result().Cookies()[0]
	h := m.Require("/auth/login", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) }))
	post := func(origin, csrf string) int {
		form := url.Values{"_csrf": {csrf}}
		r := httptest.NewRequest("POST", "http://starlight.example/console/x", strings.NewReader(form.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		if origin != "" {
			r.Header.Set("Origin", origin)
		}
		r.AddCookie(cookie)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		return rec.Code
	}
	if c := post("http://starlight.example", s.CSRF()); c != 204 {
		t.Fatalf("same-origin post = %d", c)
	}
	if c := post("https://evil.example", s.CSRF()); c != 403 {
		t.Fatalf("cross-origin post = %d", c)
	}
	if c := post("http://starlight.example", "nope"); c != 403 {
		t.Fatalf("bad csrf post = %d", c)
	}
	for _, c := range []struct {
		site, origin string
		ok           bool
	}{
		{"same-origin", "null", true}, {"same-origin", "", true}, {"cross-site", "http://starlight.example", false},
		{"same-site", "http://starlight.example", false}, {"", "null", false}, {"", "http://starlight.example", true}, {"none", "https://evil.example", false},
	} {
		r := httptest.NewRequest("POST", "http://starlight.example/console/x", nil)
		if c.site != "" {
			r.Header.Set("Sec-Fetch-Site", c.site)
		}
		if c.origin != "" {
			r.Header.Set("Origin", c.origin)
		}
		if got := session.SameOrigin(r); got != c.ok {
			t.Errorf("Sec-Fetch-Site %q Origin %q: same origin = %v", c.site, c.origin, got)
		}
	}
	anon := httptest.NewRecorder()
	h.ServeHTTP(anon, httptest.NewRequest("GET", "/console/x?a=1", nil))
	if anon.Code != http.StatusSeeOther || !strings.Contains(anon.Header().Get("Location"), "return_to=%2Fconsole%2Fx%3Fa%3D1") {
		t.Fatalf("anonymous redirect %d %s", anon.Code, anon.Header().Get("Location"))
	}
}

func TestSafeReturn(t *testing.T) {
	for in, want := range map[string]string{"/console/a": "/console/a", "//evil.example": "/c", "https://evil.example": "/c", "/\\evil": "/c", "": "/c",
		"/a\r\nx": "/c", "/a/../\\evil.example": "/c", "/%5Cevil.example": "/c", "/oauth/authorize?client_id=x&state=y": "/oauth/authorize?client_id=x&state=y"} {
		if got := session.SafeReturn(in, "/c"); got != want {
			t.Errorf("SafeReturn(%q) = %q", in, got)
		}
	}
}

func TestSAMLSessionsKeepTheNameIDAndEndByIndex(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "cy@example.com", Source: "jit"})
	var conn, other string
	for _, slug := range []string{"corp", "partner"} {
		id := ids.New()
		if _, err := pool.Exec(ctx, "INSERT INTO idp_connections (id, slug, name, protocol, preset) VALUES ($1, $2, $2, 'saml', 'generic_saml')", id, slug); err != nil {
			t.Fatal(err)
		}
		if conn == "" {
			conn = id
		} else {
			other = id
		}
	}
	m := session.New(pool, session.Config{})
	nameID := session.NameID{Format: "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent", NameQualifier: "https://idp.example.com", SPNameQualifier: "sp"}
	open := func(connID, subject, index string) *http.Cookie {
		w := httptest.NewRecorder()
		if _, err := m.Create(ctx, w, httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID, ConnectionID: connID, IDPSubject: subject,
			IDPSessionID: index, NameID: nameID}); err != nil {
			t.Fatal(err)
		}
		return w.Result().Cookies()[0]
	}
	live := func(c *http.Cookie) bool {
		r := httptest.NewRequest("GET", "/console", nil)
		r.AddCookie(c)
		_, _, err := m.Load(ctx, r)
		return err == nil
	}
	first, second, elsewhere := open(conn, "persist-cy", "idx-1"), open(conn, "persist-cy", "idx-2"), open(other, "persist-cy", "idx-1")
	r := httptest.NewRequest("GET", "/console", nil)
	r.AddCookie(first)
	s, _, err := m.Load(ctx, r)
	if err != nil || s.IDPSubject != "persist-cy" || s.IDPSessionID != "idx-1" || s.NameID != nameID {
		t.Fatalf("loaded %+v %v", s, err)
	}
	if n, err := session.RevokeSAML(ctx, pool, conn, "persist-cy", []string{"idx-1", "idx-9"}); err != nil || n != 1 {
		t.Fatalf("by index revoked %d, %v", n, err)
	}
	if live(first) || !live(second) || !live(elsewhere) {
		t.Fatal("SessionIndex logout ended the wrong sessions")
	}
	if n, _ := session.RevokeSAML(ctx, pool, conn, "persist-other", nil); n != 0 || !live(second) {
		t.Fatal("another NameID's logout ended this session")
	}
	if n, _ := session.RevokeSAML(ctx, pool, conn, "persist-cy", nil); n != 1 || live(second) || !live(elsewhere) {
		t.Fatal("NameID logout must end that connection's sessions only")
	}
	if _, err := session.RevokeSAML(ctx, pool, conn, "", nil); err == nil {
		t.Fatal("logout without a NameID accepted")
	}
}

func TestSessionsStartOnlyThroughAnEnabledConnection(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "cy@example.com", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	conn := ids.New()
	if _, err := pool.Exec(ctx, "INSERT INTO idp_connections (id, slug, name, protocol, preset) VALUES ($1, 'corp', 'Corp', 'oidc', 'generic')", conn); err != nil {
		t.Fatal(err)
	}
	m := session.New(pool, session.Config{})
	start := func(connectionID string) error {
		_, err := m.Create(ctx, httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID, ConnectionID: connectionID})
		return err
	}
	if err := start(conn); err != nil {
		t.Fatal(err)
	}
	disabling, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := disabling.Exec(ctx, "UPDATE idp_connections SET enabled = false WHERE id = $1", conn); err != nil {
		t.Fatal(err)
	}
	racing := make(chan error, 1)
	go func() { racing <- start(conn) }()
	time.Sleep(100 * time.Millisecond)
	if err := disabling.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-racing; !errors.Is(err, session.ErrConnectionOff) {
		t.Fatalf("a sign-in finishing while a disable committed = %v", err)
	}
	if _, err := pool.Exec(ctx, "DELETE FROM idp_connections WHERE id = $1", conn); err != nil {
		t.Fatal(err)
	}
	if err := start(conn); !errors.Is(err, session.ErrConnectionOff) {
		t.Fatalf("a sign-in through a deleted connection = %v", err)
	}
	if err := start(""); err != nil {
		t.Fatalf("a recovery sign-in, through no connection: %v", err)
	}
	var live int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM sessions WHERE revoked_at IS NULL").Scan(&live); err != nil || live != 2 {
		t.Fatalf("%d live sessions, want the first and the recovery one: %v", live, err)
	}
}
