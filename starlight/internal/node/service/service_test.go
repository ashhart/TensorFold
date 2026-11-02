package service

import (
	"strings"
	"testing"
)

func TestPlistRunsAsDedicatedUser(t *testing.T) {
	o := Options{Binary: "/usr/local/bin/starlight-node", Args: RunArgs("/Library/Application Support/TensorFold/Starlight Node", []string{"--engine", "/opt/a&b/tensorfold"}),
		User: "_tensorfold", StateDir: "/Library/Application Support/TensorFold/Starlight Node"}
	p, err := Plist(o)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"<string>" + Label + "</string>", "<key>UserName</key>\n\t<string>_tensorfold</string>", "<key>KeepAlive</key>\n\t<true/>",
		"<key>RunAtLoad</key>\n\t<true/>", "/opt/a&amp;b/tensorfold", "<integer>180</integer>", "/Library/Logs/TensorFold/starlight-node.log"} {
		if !strings.Contains(p, want) {
			t.Errorf("plist lacks %q", want)
		}
	}
	o.User = ""
	if _, err := Plist(o); err == nil {
		t.Fatal("system LaunchDaemon without a user")
	}
	o.Scope = "user"
	if p, err := Plist(o); err != nil || strings.Contains(p, "UserName") {
		t.Fatalf("user agent: %v", err)
	}
}

func TestUnitIsHardened(t *testing.T) {
	u, err := Unit(Options{Binary: "/usr/local/bin/starlight-node", Args: RunArgs("/var/lib/starlight-node", []string{"--hf-home", "/var/lib/starlight-node/hf cache"}),
		User: "starlight-node", StateDir: "/var/lib/starlight-node"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"User=starlight-node", "NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes", "CapabilityBoundingSet=\n",
		"StateDirectory=starlight-node", "StateDirectoryMode=0700", "TimeoutStopSec=180", "KillMode=mixed", "LimitMEMLOCK=infinity",
		`"/var/lib/starlight-node/hf cache"`} {
		if !strings.Contains(u, want) {
			t.Errorf("unit lacks %q", want)
		}
	}
	if _, err := Unit(Options{Binary: "/b", StateDir: "/var/lib/x"}); err == nil {
		t.Fatal("unit without a user")
	}
	other, _ := Unit(Options{Binary: "/b", User: "svc", StateDir: "/srv/starlight"})
	if !strings.Contains(other, "ReadWritePaths=/srv/starlight") {
		t.Fatalf("custom state dir not writable:\n%s", other)
	}
}

func TestOptionsRefuseInjection(t *testing.T) {
	bad := []Options{
		{Binary: "/b", User: "u", StateDir: "/s", Args: []string{"run\nExecStartPre=/bin/sh"}},
		{Binary: "/b", User: "u;rm", StateDir: "/s"},
		{Binary: "/b", User: "u", StateDir: "/s", Scope: "global"},
	}
	for _, o := range bad {
		if _, err := Unit(o); err == nil {
			t.Errorf("accepted %+v", o)
		}
	}
	if q := systemdQuote(`a"b$c`); q != `"a\"b$$c"` {
		t.Fatalf("quote %s", q)
	}
}
