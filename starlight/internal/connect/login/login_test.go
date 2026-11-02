package login

import (
	"bytes"
	"context"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fakestarlight"
	"github.com/ashhart/TensorFold/starlight/internal/connect/oauth"
)

func userinfo(t *testing.T, f *fakestarlight.Starlight, token string) int {
	t.Helper()
	req, _ := http.NewRequest("GET", f.Doc().UserinfoEndpoint, nil)
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

func TestBrowserPKCE(t *testing.T) {
	f := fakestarlight.New(t)
	browser := func(authURL string) error {
		u, _ := url.Parse(authURL)
		redirect := u.Query().Get("redirect_uri")
		forged, err := http.Get(redirect + "?code=stolen&state=wrong")
		if err != nil {
			return err
		}
		forged.Body.Close()
		if forged.StatusCode != http.StatusBadRequest {
			t.Errorf("forged callback answered %d", forged.StatusCode)
		}
		go func() {
			resp, err := http.Get(authURL)
			if err == nil {
				resp.Body.Close()
			}
		}()
		return nil
	}
	var out bytes.Buffer
	tr, err := Browser(context.Background(), f.Doc(), Options{Open: browser, Out: &out, DeviceName: "test-laptop", Timeout: 10 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if tr.RefreshToken == "" || userinfo(t, f, tr.AccessToken) != 200 {
		t.Fatalf("token %+v not accepted", tr)
	}
	if !strings.Contains(out.String(), "code_challenge_method=S256") || !strings.Contains(out.String(), "device_name=test-laptop") {
		t.Fatalf("printed URL missing PKCE parameters: %s", out.String())
	}
}

func TestBrowserErrorAndTimeout(t *testing.T) {
	f := fakestarlight.New(t)
	deny := func(authURL string) error {
		u, _ := url.Parse(authURL)
		q := u.Query()
		go http.Get(q.Get("redirect_uri") + "?error=access_denied&state=" + url.QueryEscape(q.Get("state")))
		return nil
	}
	if _, err := Browser(context.Background(), f.Doc(), Options{Open: deny, Timeout: 10 * time.Second}); !oauth.Is(err, "access_denied") {
		t.Fatalf("denied sign-in = %v", err)
	}
	idle := func(string) error { return nil }
	_, err := Browser(context.Background(), f.Doc(), Options{Open: idle, Timeout: 50 * time.Millisecond})
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("timeout = %v", err)
	}
}

func TestDeviceGrantPollsThroughPendingAndSlowDown(t *testing.T) {
	f := fakestarlight.New(t)
	f.DevicePending, f.DeviceSlow = 2, true
	var out bytes.Buffer
	tr, err := Device(context.Background(), f.Doc(), Options{Out: &out, unit: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	if userinfo(t, f, tr.AccessToken) != 200 || !strings.Contains(out.String(), "BCDF-GHJK") {
		t.Fatalf("device sign-in %+v, output %q", tr, out.String())
	}
}

func TestDeviceGrantExpires(t *testing.T) {
	f := fakestarlight.New(t)
	f.DevicePending = 1 << 30
	_, err := Device(context.Background(), f.Doc(), Options{unit: 50 * time.Microsecond})
	if err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("expired device grant = %v", err)
	}
}

func TestOpenBrowserRefusesOtherSchemes(t *testing.T) {
	for _, u := range []string{"file:///etc/passwd", "javascript:alert(1)", "-a Calculator"} {
		if err := OpenBrowser(u); err == nil {
			t.Errorf("opened %q", u)
		}
	}
}
