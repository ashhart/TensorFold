// Package state keeps a node's identity, credentials and trust anchors in a private directory.
package state

import (
	"crypto/ed25519"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

const (
	identityFile = "node.json"
	credsFile    = "node.pem"
	caFile       = "workload-ca.pem"
	permitFile   = "permit-keys.json"
	statusFile   = "status.json"
)

// ErrNotEnrolled means the directory holds no node identity yet.
var ErrNotEnrolled = errors.New("node is not enrolled; run `starlight-node enroll`")

// Identity is what enrolment established about this node.
type Identity struct {
	NodeID      string    `json:"node_id"`
	Name        string    `json:"name"`
	ServerURL   string    `json:"server_url"`
	TrustDomain string    `json:"trust_domain"`
	Trust       string    `json:"trust"`
	Fingerprint string    `json:"fingerprint,omitempty"`
	Address     string    `json:"address"`
	EnrolledAt  time.Time `json:"enrolled_at"`
}

// Dir is a node state directory readable only by the agent's account.
type Dir struct {
	Root string
}

// Open creates the directory if needed and tightens its permissions.
func Open(root string) (*Dir, error) {
	if root == "" {
		return nil, errors.New("state directory is empty")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if err := EnsurePrivateDir(abs); err != nil {
		return nil, fmt.Errorf("state directory %s: %w", abs, err)
	}
	return &Dir{Root: abs}, nil
}

// Path joins name under the state root.
func (d *Dir) Path(elem ...string) string {
	return filepath.Join(append([]string{d.Root}, elem...)...)
}

// Sub returns a private subdirectory, creating it when missing.
func (d *Dir) Sub(elem ...string) (string, error) {
	p := d.Path(elem...)
	return p, EnsurePrivateDir(p)
}

// Enrolled reports whether an identity exists.
func (d *Dir) Enrolled() bool {
	_, err := os.Stat(d.Path(identityFile))
	return err == nil
}

// LoadIdentity reads node.json.
func (d *Dir) LoadIdentity() (Identity, error) {
	var id Identity
	b, err := os.ReadFile(d.Path(identityFile))
	if errors.Is(err, os.ErrNotExist) {
		return id, ErrNotEnrolled
	}
	if err != nil {
		return id, err
	}
	if err := json.Unmarshal(b, &id); err != nil {
		return id, fmt.Errorf("%s: %w", identityFile, err)
	}
	if id.NodeID == "" || id.ServerURL == "" || id.TrustDomain == "" {
		return id, fmt.Errorf("%s is incomplete; re-enrol the node", identityFile)
	}
	return id, nil
}

// SaveIdentity writes node.json atomically.
func (d *Dir) SaveIdentity(id Identity) error {
	b, err := json.MarshalIndent(id, "", "  ")
	if err != nil {
		return err
	}
	return WriteFileAtomic(d.Path(identityFile), append(b, '\n'), 0o600)
}

// SaveCredentials replaces the certificate and key together in one atomic rename.
func (d *Dir) SaveCredentials(certPEM, keyPEM []byte) error {
	combined := append(append([]byte{}, certPEM...), keyPEM...)
	return WriteFileAtomic(d.Path(credsFile), combined, 0o600)
}

// CredentialsPEM returns the combined certificate and key PEM.
func (d *Dir) CredentialsPEM() ([]byte, error) {
	b, err := os.ReadFile(d.Path(credsFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotEnrolled
	}
	return b, err
}

// SaveWorkloadCA pins the internal CA that signs gateway and node certificates.
func (d *Dir) SaveWorkloadCA(certPEM []byte) error {
	return WriteFileAtomic(d.Path(caFile), certPEM, 0o600)
}

// WorkloadCA loads the pinned internal CA.
func (d *Dir) WorkloadCA() (*x509.CertPool, *x509.Certificate, error) {
	return loadPool(d.Path(caFile))
}

func loadPool(path string) (*x509.CertPool, *x509.Certificate, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	pool := x509.NewCertPool()
	var first *x509.Certificate
	for len(b) > 0 {
		var block *pem.Block
		block, b = pem.Decode(b)
		if block == nil {
			break
		}
		if block.Type != "CERTIFICATE" {
			continue
		}
		c, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", path, err)
		}
		pool.AddCert(c)
		if first == nil {
			first = c
		}
	}
	if first == nil {
		return nil, nil, fmt.Errorf("%s holds no certificate", path)
	}
	return pool, first, nil
}

// SavePermitKeys stores the gateway permit keys so a restarted agent verifies before its first sync.
func (d *Dir) SavePermitKeys(keys []wire.PermitKey) error {
	b, err := json.Marshal(keys)
	if err != nil {
		return err
	}
	return WriteFileAtomic(d.Path(permitFile), b, 0o600)
}

// PermitKeys loads the stored permit keys as a verifier key map.
func (d *Dir) PermitKeys() (map[string]ed25519.PublicKey, error) {
	b, err := os.ReadFile(d.Path(permitFile))
	if errors.Is(err, os.ErrNotExist) {
		return map[string]ed25519.PublicKey{}, nil
	}
	if err != nil {
		return nil, err
	}
	var keys []wire.PermitKey
	if err := json.Unmarshal(b, &keys); err != nil {
		return nil, err
	}
	return KeyMap(keys), nil
}

// KeyMap converts wire keys to a verifier map, dropping malformed entries.
func KeyMap(keys []wire.PermitKey) map[string]ed25519.PublicKey {
	out := make(map[string]ed25519.PublicKey, len(keys))
	for _, k := range keys {
		if k.KID != "" && len(k.PublicKey) == ed25519.PublicKeySize {
			out[k.KID] = ed25519.PublicKey(k.PublicKey)
		}
	}
	return out
}

// WriteStatus records the agent's latest view for `starlight-node status`.
func (d *Dir) WriteStatus(v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return WriteFileAtomic(d.Path(statusFile), append(b, '\n'), 0o600)
}

// ReadStatus returns the raw status written by a running agent.
func (d *Dir) ReadStatus() ([]byte, error) {
	return os.ReadFile(d.Path(statusFile))
}

// Forget removes identity, credentials and keys so the node can enrol again.
func (d *Dir) Forget() error {
	for _, f := range []string{identityFile, credsFile, caFile, permitFile, statusFile} {
		if err := os.Remove(d.Path(f)); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}
