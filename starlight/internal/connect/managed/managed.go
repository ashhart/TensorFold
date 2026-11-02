// Package managed reads the configuration an administrator pushes through MDM or group policy.
package managed

import (
	"bytes"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Domain is the macOS preference domain and the name used in every platform's policy location.
const Domain = "dev.tensorfold.connect"

// Policy is managed configuration; any field it sets overrides and locks the user's own setting.
type Policy struct {
	ServerURL         string   `json:"ServerURL"`
	CACertificatePEM  string   `json:"CACertificatePEM"`
	CAFile            string   `json:"CAFile"`
	AllowManualServer *bool    `json:"AllowManualServer"`
	BridgePort        int      `json:"BridgePort"`
	DisableBridge     bool     `json:"DisableBridge"`
	Sources           []string `json:"-"`
	Unknown           []string `json:"-"`
}

// Empty reports whether no policy applies.
func (p Policy) Empty() bool {
	return p.ServerURL == "" && p.CACertificatePEM == "" && p.CAFile == "" && p.AllowManualServer == nil && p.BridgePort == 0 && !p.DisableBridge
}

// ServerLocked reports whether the user may not choose a different server.
func (p Policy) ServerLocked() bool {
	return p.ServerURL != "" && (p.AllowManualServer == nil || !*p.AllowManualServer)
}

// Validate refuses values Connect could not apply safely.
func (p Policy) Validate() error {
	if p.BridgePort != 0 && (p.BridgePort < 1024 || p.BridgePort > 65535) {
		return fmt.Errorf("managed BridgePort %d must be between 1024 and 65535", p.BridgePort)
	}
	if p.CACertificatePEM != "" {
		if _, err := ParseCAs([]byte(p.CACertificatePEM)); err != nil {
			return fmt.Errorf("managed CACertificatePEM: %w", err)
		}
	}
	return nil
}

// ParseCAs decodes every PEM certificate in data and refuses input that holds none.
func ParseCAs(data []byte) ([]*x509.Certificate, error) {
	var out []*x509.Certificate
	rest := bytes.TrimSpace(data)
	for len(rest) > 0 {
		var block *pem.Block
		block, rest = pem.Decode(rest)
		if block == nil {
			return nil, errors.New("not PEM encoded")
		}
		if block.Type != "CERTIFICATE" {
			continue
		}
		c, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
		rest = bytes.TrimSpace(rest)
	}
	if len(out) == 0 {
		return nil, errors.New("no certificates found")
	}
	return out, nil
}

// Load reads the platform's managed configuration; a missing policy is not an error.
func Load() (Policy, error) {
	p, err := loadPlatform()
	if err != nil {
		return Policy{}, err
	}
	return p, p.Validate()
}

// Merge overlays later on earlier, so a more authoritative source passed last wins field by field.
func Merge(earlier, later Policy) Policy {
	out := earlier
	if later.ServerURL != "" {
		out.ServerURL = later.ServerURL
	}
	if later.CACertificatePEM != "" {
		out.CACertificatePEM = later.CACertificatePEM
	}
	if later.CAFile != "" {
		out.CAFile = later.CAFile
	}
	if later.AllowManualServer != nil {
		v := *later.AllowManualServer
		out.AllowManualServer = &v
	}
	if later.BridgePort != 0 {
		out.BridgePort = later.BridgePort
	}
	if later.DisableBridge {
		out.DisableBridge = true
	}
	out.Sources = append(append([]string(nil), earlier.Sources...), later.Sources...)
	out.Unknown = append(append([]string(nil), earlier.Unknown...), later.Unknown...)
	return out
}

// ParseJSON decodes a policy document; unknown keys are kept as warnings and MDM Payload keys are ignored.
func ParseJSON(data []byte, source string) (Policy, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return Policy{}, fmt.Errorf("managed configuration %s: %w", source, err)
	}
	var p Policy
	fields := map[string]any{"ServerURL": &p.ServerURL, "CACertificatePEM": &p.CACertificatePEM, "CAFile": &p.CAFile,
		"AllowManualServer": &p.AllowManualServer, "BridgePort": &p.BridgePort, "DisableBridge": &p.DisableBridge}
	for key, value := range raw {
		dst, ok := fields[key]
		if !ok {
			if !strings.HasPrefix(key, "Payload") {
				p.Unknown = append(p.Unknown, key)
			}
			continue
		}
		if err := json.Unmarshal(value, dst); err != nil {
			return Policy{}, fmt.Errorf("managed configuration %s: %s: %w", source, key, err)
		}
	}
	sort.Strings(p.Unknown)
	if !p.Empty() {
		p.Sources = []string{source}
	}
	return p, nil
}

// LoadJSONFiles reads each existing file in order and merges them, later files winning.
func LoadJSONFiles(paths ...string) (Policy, error) {
	var out Policy
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return Policy{}, fmt.Errorf("managed configuration %s: %w", path, err)
		}
		p, err := ParseJSON(data, path)
		if err != nil {
			return Policy{}, err
		}
		out = Merge(out, p)
	}
	return out, nil
}
