package web

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRenderEscapesAndSetsHeaders(t *testing.T) {
	r, err := New(func(context.Context) string { return "Example <Corp>" })
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	page := LoginPage{ShowEmail: true, ReturnTo: "/console?a=1&b=<x>", Error: "<script>alert(1)</script>",
		Options: []LoginOption{{Slug: "corp", Name: "Corp & Co"}}}
	r.Render(w, httptest.NewRequest("GET", "/auth/login", nil), http.StatusOK, PageLogin, "Sign in", page)
	body := w.Body.String()
	if strings.Contains(body, "<script>alert") || !strings.Contains(body, "&lt;script&gt;") {
		t.Fatal("error text not escaped")
	}
	if !strings.Contains(body, "Example &lt;Corp&gt;") || !strings.Contains(body, "Corp &amp; Co") {
		t.Fatal("org or option not escaped")
	}
	if !strings.Contains(body, "return_to=%2fconsole%3fa%3d1%26b%3d%3cx%3e") {
		t.Fatalf("return_to not query-escaped: %s", body)
	}
	csp := w.Header().Get("Content-Security-Policy")
	if csp != CSP || strings.Contains(csp, "form-action") {
		t.Fatalf("csp %q", csp)
	}
}

func TestEveryPageRenders(t *testing.T) {
	r, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]any{
		PageLogin:   LoginPage{SetupCode: "slx_x", Options: []LoginOption{{Slug: "a", Name: "A"}}},
		PageMessage: MessagePage{Kind: "error", Heading: "No", Body: "Nope", Links: []Link{{Href: "/auth/login", Label: "Back"}}},
		PageApprove: ApprovePage{ClientName: "Connect", UserEmail: "a@example.com", Scopes: []string{"inference"}, Action: "/oauth/authorize",
			Hidden: []Hidden{{Name: "state", Value: "s"}}},
		PageDevice:  DevicePage{Action: "/device", Confirm: true, UserCode: "BCDF-GHJK"},
		PageSetup:   SetupPage{Step: "configure", Protocol: "oidc", Fields: []FormField{{Name: "x", Kind: "select", Options: []string{"a"}}, {Name: "y", Kind: "bool"}}},
		PageRecover: RecoverPage{Code: "slx_y"},
	}
	for page, data := range cases {
		w := httptest.NewRecorder()
		r.Render(w, httptest.NewRequest("GET", "/", nil), 200, page, "T", data)
		if w.Code != 200 || !strings.Contains(w.Body.String(), "TensorFold Starlight") {
			t.Errorf("%s: %d %s", page, w.Code, w.Body.String())
		}
	}
}

func TestStaticHandler(t *testing.T) {
	h := StaticHandler()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", Asset("starlight.css"), nil))
	if w.Code != 200 || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/css") || !strings.Contains(w.Header().Get("Cache-Control"), "immutable") {
		t.Fatalf("css %d %v", w.Code, w.Header())
	}
	for _, bad := range []string{"/static/../web.go", "/static/missing.css", "/static/"} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", bad, nil))
		if w.Code != 404 {
			t.Errorf("%s = %d", bad, w.Code)
		}
	}
}
