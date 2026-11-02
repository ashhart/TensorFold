// Package profile stores the user's Connect settings and resolves them against managed policy.
package profile

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fsutil"
	"github.com/ashhart/TensorFold/starlight/internal/connect/managed"
)

// HomeEnv overrides the configuration directory, for tests and for keeping separate profiles.
const HomeEnv = "TENSORFOLD_CONNECT_HOME"

// DefaultBridgePort is where the local endpoint listens unless policy or the user says otherwise.
const DefaultBridgePort = 7171

// Dir returns the per-user configuration directory.
func Dir() (string, error) {
	if d := os.Getenv(HomeEnv); d != "" {
		return filepath.Abs(d)
	}
	base, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "tensorfold-connect"), nil
}

// Profile is what the user chose; managed policy can override every field.
type Profile struct {
	Server         string   `json:"server,omitempty"`
	Project        string   `json:"project,omitempty"`
	BridgePort     int      `json:"bridge_port,omitempty"`
	CAFile         string   `json:"ca_file,omitempty"`
	AllowedOrigins []string `json:"allowed_origins,omitempty"`
}

func path(dir string) string { return filepath.Join(dir, "profile.json") }

// Load reads the profile; a missing file is an empty profile.
func Load(dir string) (Profile, error) {
	var p Profile
	data, err := os.ReadFile(path(dir))
	if errors.Is(err, os.ErrNotExist) {
		return p, nil
	}
	if err != nil {
		return p, err
	}
	if err := json.Unmarshal(data, &p); err != nil {
		return p, fmt.Errorf("profile: %w", err)
	}
	return p, nil
}

// Save writes the profile readable only by the user.
func Save(dir string, p Profile) error {
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return fsutil.WriteAtomic(path(dir), append(data, '\n'), 0o600)
}

// Settings are the effective values after managed policy is applied.
type Settings struct {
	Server         string
	ServerManaged  bool
	ServerLocked   bool
	CAPEM          []byte
	CASource       string
	Project        string
	BridgePort     int
	BridgeDisabled bool
	AllowedOrigins []string
	Policy         managed.Policy
}

// Resolve applies policy over the profile and loads the trust anchors either names.
func Resolve(p Profile, m managed.Policy) (Settings, error) {
	s := Settings{Server: p.Server, Project: p.Project, BridgePort: p.BridgePort, AllowedOrigins: p.AllowedOrigins, Policy: m}
	if m.ServerURL != "" {
		s.Server, s.ServerManaged, s.ServerLocked = m.ServerURL, true, m.ServerLocked()
		if !s.ServerLocked && p.Server != "" {
			s.Server = p.Server
		}
	}
	if s.Server != "" {
		norm, err := NormalizeServer(s.Server)
		if err != nil {
			return s, err
		}
		s.Server = norm
	}
	if m.BridgePort != 0 {
		s.BridgePort = m.BridgePort
	}
	if s.BridgePort == 0 {
		s.BridgePort = DefaultBridgePort
	}
	s.BridgeDisabled = m.DisableBridge
	switch {
	case m.CACertificatePEM != "":
		s.CAPEM, s.CASource = []byte(m.CACertificatePEM), "managed policy"
	case m.CAFile != "":
		data, err := os.ReadFile(m.CAFile)
		if err != nil {
			return s, fmt.Errorf("managed CAFile: %w", err)
		}
		s.CAPEM, s.CASource = data, m.CAFile
	case p.CAFile != "":
		data, err := os.ReadFile(p.CAFile)
		if err != nil {
			return s, fmt.Errorf("profile ca_file: %w", err)
		}
		s.CAPEM, s.CASource = data, p.CAFile
	}
	if len(s.CAPEM) > 0 {
		if _, err := managed.ParseCAs(s.CAPEM); err != nil {
			return s, fmt.Errorf("CA from %s: %w", s.CASource, err)
		}
	}
	return s, nil
}

// NormalizeServer checks a Starlight URL: HTTPS, or plain HTTP only on loopback, with no query or credentials.
func NormalizeServer(raw string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") {
		return "", fmt.Errorf("server %q must be an absolute https URL", raw)
	}
	if u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("server %q must not carry credentials, a query or a fragment", raw)
	}
	if u.Scheme == "http" && !Loopback(u.Hostname()) {
		return "", fmt.Errorf("server %q must use https; plain http is allowed only on this machine", raw)
	}
	u.Path = strings.TrimRight(u.Path, "/")
	return u.String(), nil
}

// Loopback reports whether host names this machine.
func Loopback(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}
