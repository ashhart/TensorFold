package httpx

import (
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"
)

func TestClientIPHonoursOnlyTrustedHops(t *testing.T) {
	p, err := NewProxies([]string{"10.0.0.0/8"})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		remote, xff, want string
	}{
		{"203.0.113.5:1234", "198.51.100.1", "203.0.113.5"},
		{"10.0.0.2:1234", "198.51.100.1", "198.51.100.1"},
		{"10.0.0.2:1234", "198.51.100.9, 198.51.100.1, 10.0.0.3", "198.51.100.1"},
		{"10.0.0.2:1234", "", "10.0.0.2"},
		{"[::ffff:10.0.0.2]:1234", "198.51.100.7", "198.51.100.7"},
		{"10.0.0.2:1234", "203.0.113.7:51544", "203.0.113.7"},
		{"10.0.0.2:1234", "198.51.100.9, 10.0.0.3:443", "198.51.100.9"},
		{"10.0.0.2:1234", "[2001:db8::1]:443", "2001:db8::1"},
		{"10.0.0.2:1234", "[::ffff:198.51.100.4]:80", "198.51.100.4"},
		{"10.0.0.2:1234", "evil, 203.0.113.7:1", "203.0.113.7"},
	}
	for _, c := range cases {
		r := httptest.NewRequest("GET", "/", nil)
		r.RemoteAddr = c.remote
		if c.xff != "" {
			r.Header.Set("X-Forwarded-For", c.xff)
		}
		if got := p.ClientIP(r).String(); got != c.want {
			t.Errorf("%s via %q = %s, want %s", c.remote, c.xff, got, c.want)
		}
	}
}

func TestClientIPUnreadableHopIsNoAddress(t *testing.T) {
	p, err := NewProxies([]string{"10.0.0.0/8"})
	if err != nil {
		t.Fatal(err)
	}
	everything := []netip.Prefix{netip.MustParsePrefix("0.0.0.0/0"), netip.MustParsePrefix("::/0"), netip.MustParsePrefix("10.0.0.0/8")}
	for _, xff := range []string{"garbage", "203.0.113.7, unknown", "198.51.100.1, ", "203.0.113.7:99999", "[2001:db8::1]"} {
		r := httptest.NewRequest("GET", "/", nil)
		r.RemoteAddr = "10.0.0.2:1234"
		r.Header.Set("X-Forwarded-For", xff)
		got := p.ClientIP(r)
		if got.IsValid() {
			t.Errorf("%q = %s, want no address rather than a guess", xff, got)
		}
		for _, allow := range everything {
			if allow.Contains(got) {
				t.Errorf("%q matched the allowlist %s", xff, allow)
			}
		}
	}
}

func TestBearerToken(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.Header.Set("Authorization", "bearer abc")
	if BearerToken(r) != "abc" {
		t.Fatal("lowercase bearer not accepted")
	}
	r.Header.Set("Authorization", "Basic abc")
	r.Header.Set("X-Api-Key", "k")
	if BearerToken(r) != "" {
		t.Fatal("basic auth must not fall through to x-api-key")
	}
	r.Header.Del("Authorization")
	if BearerToken(r) != "k" {
		t.Fatal("x-api-key not read")
	}
}

func TestWriteDialects(t *testing.T) {
	w := httptest.NewRecorder()
	Write(w, Anthropic, Errorf(http.StatusServiceUnavailable, "capacity", "busy"))
	if !strings.Contains(w.Body.String(), `"overloaded_error"`) {
		t.Fatalf("anthropic body %s", w.Body)
	}
	w = httptest.NewRecorder()
	e := Errorf(http.StatusTooManyRequests, "concurrency_limit", "slow down")
	e.RetryAfter = 2
	Write(w, OpenAI, e)
	if w.Header().Get("Retry-After") != "2" || !strings.Contains(w.Body.String(), `"rate_limit_error"`) {
		t.Fatalf("openai %v %s", w.Header(), w.Body)
	}
	w = httptest.NewRecorder()
	Write(w, OpenAI, http.ErrAbortHandler)
	if w.Code != 500 || strings.Contains(w.Body.String(), "abort") {
		t.Fatal("internal error leaked")
	}
}
