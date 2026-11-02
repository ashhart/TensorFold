//go:build darwin

package managed

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadPlistsUserOverridesComputer(t *testing.T) {
	dir := t.TempDir()
	computer := filepath.Join(dir, Domain+".plist")
	user := filepath.Join(dir, "user", Domain+".plist")
	os.MkdirAll(filepath.Dir(user), 0o755)
	write := func(path, body string) {
		doc := `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>` + body + `</dict></plist>`
		if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(computer, `<key>ServerURL</key><string>https://starlight.example</string><key>BridgePort</key><integer>7400</integer>
		<key>AllowManualServer</key><false/>`)
	write(user, `<key>DisableBridge</key><true/>`)
	p, err := loadPlists(computer, user)
	if err != nil {
		t.Fatal(err)
	}
	if p.ServerURL != "https://starlight.example" || p.BridgePort != 7400 || !p.DisableBridge || !p.ServerLocked() || len(p.Sources) != 2 {
		t.Fatalf("policy %+v", p)
	}
}

func TestShippedPlistSampleParses(t *testing.T) {
	p, err := loadPlists(filepath.Join("..", "..", "..", "deploy", "mdm", "macos", Domain+".plist"))
	if err != nil {
		t.Fatal(err)
	}
	if p.ServerURL == "" || len(p.Unknown) != 0 || p.BridgePort != 7171 || !p.ServerLocked() {
		t.Fatalf("sample policy %+v", p)
	}
}
