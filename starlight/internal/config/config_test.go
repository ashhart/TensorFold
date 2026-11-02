package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAppliesFileThenEnv(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "starlight.yaml")
	body := "public_url: https://ai.example.com\nroles: [gateway]\ngateway:\n  default_max_tokens: 2048\n  queue_batch: 5m\n"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("STARLIGHT_DATABASE_URL", "postgres://starlight@db.example/starlight")
	t.Setenv("STARLIGHT_ROLES", "gateway,control")
	c, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if c.Gateway.DefaultMaxTokens != 2048 || c.Gateway.QueueBatch.Minutes() != 5 || !c.HasRole("control") || !c.Secure() {
		t.Fatalf("config = %+v", c)
	}
	if u, _ := c.DatabaseURL(); u != "postgres://starlight@db.example/starlight" {
		t.Fatalf("database url %q", u)
	}
}

func TestValidateRefuses(t *testing.T) {
	cases := map[string]func(*Config){
		"plain http":      func(c *Config) { c.PublicURL = "http://ai.example.com" },
		"path":            func(c *Config) { c.PublicURL = "https://ai.example.com/starlight" },
		"role":            func(c *Config) { c.Roles = []string{"master"} },
		"half tls":        func(c *Config) { c.TLS.CertFile = "a.pem" },
		"cache too long":  func(c *Config) { c.Sessions.AuthCache.Duration *= 10 },
		"tiny body limit": func(c *Config) { c.Gateway.MaxBodyBytes = 10 },
		"huge max tokens": func(c *Config) { c.Gateway.DefaultMaxTokens = 1 << 40 },
	}
	for name, mutate := range cases {
		c := Defaults()
		c.PublicURL = "https://ai.example.com"
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	c := Defaults()
	c.PublicURL = "http://127.0.0.1:8443"
	if err := c.Validate(); err != nil {
		t.Fatalf("loopback http refused: %v", err)
	}
}

func TestUnknownFieldsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "starlight.yaml")
	_ = os.WriteFile(path, []byte("public_url: https://ai.example.com\npublic_ulr: typo\n"), 0o600)
	if _, err := Load(path); err == nil {
		t.Fatal("typo accepted")
	}
}
