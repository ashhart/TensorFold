package profile

import (
	"os"
	"runtime"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/connect/managed"
)

func TestSaveLoadPrivate(t *testing.T) {
	dir := t.TempDir() + "/connect"
	want := Profile{Server: "https://starlight.example", Project: "research", BridgePort: 7300}
	if err := Save(dir, want); err != nil {
		t.Fatal(err)
	}
	got, err := Load(dir)
	if err != nil || got.Server != want.Server || got.Project != want.Project || got.BridgePort != 7300 {
		t.Fatalf("load = %+v, %v", got, err)
	}
	if runtime.GOOS != "windows" {
		fi, _ := os.Stat(dir + "/profile.json")
		if fi.Mode().Perm() != 0o600 {
			t.Fatalf("profile mode %v", fi.Mode().Perm())
		}
	}
	empty, err := Load(t.TempDir())
	if err != nil || empty.Server != "" {
		t.Fatalf("missing profile = %+v, %v", empty, err)
	}
}

func TestResolveManagedOverridesUser(t *testing.T) {
	allow := true
	user := Profile{Server: "https://mine.example/", BridgePort: 7300}
	s, err := Resolve(user, managed.Policy{ServerURL: "https://corp.example", BridgePort: 7500, DisableBridge: true})
	if err != nil {
		t.Fatal(err)
	}
	if s.Server != "https://corp.example" || !s.ServerLocked || s.BridgePort != 7500 || !s.BridgeDisabled {
		t.Fatalf("locked settings %+v", s)
	}
	s, err = Resolve(user, managed.Policy{ServerURL: "https://corp.example", AllowManualServer: &allow})
	if err != nil || s.Server != "https://mine.example" || s.ServerLocked || !s.ServerManaged {
		t.Fatalf("manual override %+v, %v", s, err)
	}
	s, _ = Resolve(Profile{}, managed.Policy{})
	if s.BridgePort != DefaultBridgePort || s.Server != "" {
		t.Fatalf("defaults %+v", s)
	}
}

func TestNormalizeServer(t *testing.T) {
	ok := map[string]string{
		"https://starlight.example/":     "https://starlight.example",
		" https://starlight.example/ai ": "https://starlight.example/ai",
		"http://127.0.0.1:8443":          "http://127.0.0.1:8443",
		"http://localhost:9000/":         "http://localhost:9000",
		"http://[::1]:9000":              "http://[::1]:9000",
	}
	for in, want := range ok {
		got, err := NormalizeServer(in)
		if err != nil || got != want {
			t.Errorf("NormalizeServer(%q) = %q, %v", in, got, err)
		}
	}
	for _, bad := range []string{"starlight.example", "http://starlight.example", "https://u:p@starlight.example", "https://starlight.example?x=1", "ftp://starlight.example"} {
		if _, err := NormalizeServer(bad); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
}
