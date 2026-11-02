// Package config loads Starlight's YAML configuration with environment overrides and validates it.
package config

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"strings"
	"time"

	"go.yaml.in/yaml/v3"
)

// Duration unmarshals from Go duration strings such as "30s".
type Duration struct{ time.Duration }

// UnmarshalYAML parses a duration string.
func (d *Duration) UnmarshalYAML(n *yaml.Node) error {
	v, err := time.ParseDuration(n.Value)
	if err != nil {
		return fmt.Errorf("line %d: %w", n.Line, err)
	}
	d.Duration = v
	return nil
}

// Config is the whole server configuration.
type Config struct {
	PublicURL      string   `yaml:"public_url"`
	NodeURL        string   `yaml:"node_url"`
	Roles          []string `yaml:"roles"`
	TrustedProxies []string `yaml:"trusted_proxies"`
	Listen         Listen   `yaml:"listen"`
	TLS            TLS      `yaml:"tls"`
	Database       Database `yaml:"database"`
	Keys           Keys     `yaml:"keys"`
	Identity       Identity `yaml:"identity"`
	Sessions       Sessions `yaml:"sessions"`
	Gateway        Gateway  `yaml:"gateway"`
	MDNS           MDNS     `yaml:"mdns"`
	Metrics        Metrics  `yaml:"metrics"`
	Log            Log      `yaml:"log"`
}

// Listen holds the three listener addresses.
type Listen struct {
	Public  string `yaml:"public"`
	Nodes   string `yaml:"nodes"`
	Metrics string `yaml:"metrics"`
}

// TLS configures the public listener; empty files mean TLS terminates at a proxy.
type TLS struct {
	CertFile string `yaml:"cert_file"`
	KeyFile  string `yaml:"key_file"`
}

// Database locates Postgres.
type Database struct {
	URL      string `yaml:"url"`
	URLFile  string `yaml:"url_file"`
	MaxConns int32  `yaml:"max_conns"`
	MinConns int32  `yaml:"min_conns"`
}

// Keys locates the master key; previous keys only unwrap during rotation.
type Keys struct {
	MasterKeyFile    string   `yaml:"master_key_file"`
	PreviousKeyFiles []string `yaml:"previous_key_files"`
	MasterKey        string   `yaml:"-"`
}

// Identity configures how Starlight reaches identity providers.
type Identity struct {
	CAFile string `yaml:"ca_file"`
}

// Sessions bounds browser sessions and Connect tokens.
type Sessions struct {
	ConsoleIdle     Duration `yaml:"console_idle"`
	ConsoleAbsolute Duration `yaml:"console_absolute"`
	AccessTTL       Duration `yaml:"access_ttl"`
	RefreshIdle     Duration `yaml:"refresh_idle"`
	RefreshAbsolute Duration `yaml:"refresh_absolute"`
	AuthCache       Duration `yaml:"auth_cache"`
}

// Gateway tunes the inference path.
type Gateway struct {
	MaxBodyBytes     int64    `yaml:"max_body_bytes"`
	QueueInteractive Duration `yaml:"queue_interactive"`
	QueueStandard    Duration `yaml:"queue_standard"`
	QueueBatch       Duration `yaml:"queue_batch"`
	RequestTimeout   Duration `yaml:"request_timeout"`
	DefaultMaxTokens int      `yaml:"default_max_tokens"`
	AgeingStep       Duration `yaml:"ageing_step"`
}

// MDNS controls LAN advertisement of the public URL.
type MDNS struct {
	Advertise bool   `yaml:"advertise"`
	Instance  string `yaml:"instance"`
}

// Metrics protects the metrics listener.
type Metrics struct {
	BearerTokenFile string `yaml:"bearer_token_file"`
}

// Log selects level and format.
type Log struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

// Defaults returns a configuration that validates once PublicURL and a database are set.
func Defaults() Config {
	return Config{
		Roles:    []string{"gateway", "control", "console", "scim"},
		Listen:   Listen{Public: ":8443", Nodes: ":9443", Metrics: "127.0.0.1:9090"},
		Database: Database{MinConns: 4},
		Sessions: Sessions{
			ConsoleIdle: Duration{30 * time.Minute}, ConsoleAbsolute: Duration{12 * time.Hour}, AccessTTL: Duration{10 * time.Minute},
			RefreshIdle: Duration{30 * 24 * time.Hour}, RefreshAbsolute: Duration{90 * 24 * time.Hour}, AuthCache: Duration{15 * time.Second},
		},
		Gateway: Gateway{
			MaxBodyBytes: 32 << 20, QueueInteractive: Duration{30 * time.Second}, QueueStandard: Duration{2 * time.Minute},
			QueueBatch: Duration{15 * time.Minute}, RequestTimeout: Duration{30 * time.Minute}, DefaultMaxTokens: 4096,
			AgeingStep: Duration{30 * time.Second},
		},
		Log: Log{Level: "info", Format: "json"},
	}
}

// Load reads path (optional) over the defaults, then applies STARLIGHT_* environment overrides.
func Load(path string) (Config, error) {
	c := Defaults()
	if path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			return c, err
		}
		dec := yaml.NewDecoder(strings.NewReader(string(raw)))
		dec.KnownFields(true)
		if err := dec.Decode(&c); err != nil && !errors.Is(err, io.EOF) {
			return c, fmt.Errorf("%s: %w", path, err)
		}
	}
	c.applyEnv(os.Getenv)
	return c, c.Validate()
}

func (c *Config) applyEnv(get func(string) string) {
	set := func(dst *string, key string) {
		if v := get(key); v != "" {
			*dst = v
		}
	}
	set(&c.PublicURL, "STARLIGHT_PUBLIC_URL")
	set(&c.NodeURL, "STARLIGHT_NODE_URL")
	set(&c.Listen.Public, "STARLIGHT_LISTEN_PUBLIC")
	set(&c.Listen.Nodes, "STARLIGHT_LISTEN_NODES")
	set(&c.Listen.Metrics, "STARLIGHT_LISTEN_METRICS")
	set(&c.TLS.CertFile, "STARLIGHT_TLS_CERT_FILE")
	set(&c.TLS.KeyFile, "STARLIGHT_TLS_KEY_FILE")
	set(&c.Database.URL, "STARLIGHT_DATABASE_URL")
	set(&c.Database.URLFile, "STARLIGHT_DATABASE_URL_FILE")
	set(&c.Keys.MasterKey, "STARLIGHT_MASTER_KEY")
	set(&c.Keys.MasterKeyFile, "STARLIGHT_MASTER_KEY_FILE")
	set(&c.Log.Level, "STARLIGHT_LOG_LEVEL")
	set(&c.Identity.CAFile, "STARLIGHT_IDP_CA_FILE")
	if v := get("STARLIGHT_ROLES"); v != "" {
		c.Roles = strings.Split(v, ",")
	}
	if v := get("STARLIGHT_TRUSTED_PROXIES"); v != "" {
		c.TrustedProxies = strings.Split(v, ",")
	}
}

// DatabaseURL resolves the URL from the value or its file.
func (c Config) DatabaseURL() (string, error) {
	if c.Database.URL != "" {
		return c.Database.URL, nil
	}
	if c.Database.URLFile != "" {
		b, err := os.ReadFile(c.Database.URLFile)
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(string(b)), nil
	}
	return "", errors.New("no database: set database.url, database.url_file or STARLIGHT_DATABASE_URL")
}

// HasRole reports whether this instance runs role.
func (c Config) HasRole(role string) bool {
	for _, r := range c.Roles {
		if strings.TrimSpace(r) == role {
			return true
		}
	}
	return false
}

// Secure reports whether browsers reach Starlight over HTTPS.
func (c Config) Secure() bool {
	return strings.HasPrefix(c.PublicURL, "https://")
}

// Validate refuses configurations that would start insecurely or ambiguously.
func (c Config) Validate() error {
	u, err := url.Parse(c.PublicURL)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" || u.Path != "" && u.Path != "/" {
		return fmt.Errorf("public_url must be an absolute http(s) origin without a path, got %q", c.PublicURL)
	}
	if u.Scheme == "http" && !loopbackHost(u.Hostname()) {
		return errors.New("public_url must use https unless it is a loopback address")
	}
	if c.NodeURL != "" {
		n, err := url.Parse(c.NodeURL)
		if err != nil || n.Scheme != "https" || n.Host == "" {
			return fmt.Errorf("node_url must be an https origin, got %q", c.NodeURL)
		}
	}
	for _, r := range c.Roles {
		switch strings.TrimSpace(r) {
		case "gateway", "control", "console", "scim":
		default:
			return fmt.Errorf("unknown role %q", r)
		}
	}
	if (c.TLS.CertFile == "") != (c.TLS.KeyFile == "") {
		return errors.New("tls.cert_file and tls.key_file must be set together")
	}
	for _, a := range []string{c.Listen.Public, c.Listen.Nodes, c.Listen.Metrics} {
		if _, _, err := net.SplitHostPort(a); err != nil {
			return fmt.Errorf("listen address %q: %w", a, err)
		}
	}
	if c.Sessions.AccessTTL.Duration < time.Minute || c.Sessions.AccessTTL.Duration > time.Hour {
		return errors.New("sessions.access_ttl must be between 1m and 1h")
	}
	if c.Sessions.AuthCache.Duration < 0 || c.Sessions.AuthCache.Duration > time.Minute {
		return errors.New("sessions.auth_cache must be at most 1m, the revocation bound")
	}
	if c.Gateway.MaxBodyBytes < 1<<20 || c.Gateway.MaxBodyBytes > 512<<20 {
		return errors.New("gateway.max_body_bytes must be between 1 MiB and 512 MiB")
	}
	if c.Gateway.DefaultMaxTokens < 1 || c.Gateway.DefaultMaxTokens > 1<<20 {
		return errors.New("gateway.default_max_tokens must be between 1 and 1048576")
	}
	return nil
}

func loopbackHost(h string) bool {
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}
