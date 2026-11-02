// Package enroll turns a one-time token into a node identity and keeps its certificate fresh.
package enroll

import (
	"context"
	"crypto/ecdsa"
	"crypto/sha256"
	"crypto/subtle"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/controlclient"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
	"github.com/ashhart/TensorFold/starlight/internal/node/trust"
)

var namePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$`)

// Options configure one enrolment.
type Options struct {
	Server   string
	Token    string
	Name     string
	Address  string
	CABundle []byte
	CAPin    string
	Facts    wire.Facts
	Force    bool
}

// Mode picks how enrolment authenticates Starlight: a pinned CA, a CA bundle, or the system roots.
func (o Options) Mode() (trust.Mode, error) {
	switch {
	case o.CAPin != "" && len(o.CABundle) > 0:
		return trust.Mode{}, errors.New("use either --ca-file or --ca-sha256, not both")
	case o.CAPin != "":
		pin, err := trust.ParsePin(o.CAPin)
		if err != nil {
			return trust.Mode{}, err
		}
		return trust.Mode{Kind: trust.CAPin, Pin: pin}, nil
	case len(o.CABundle) > 0:
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(o.CABundle) {
			return trust.Mode{}, errors.New("--ca-file holds no PEM certificate")
		}
		return trust.Mode{Kind: trust.CAFile, Pool: pool}, nil
	default:
		return trust.Mode{Kind: trust.System}, nil
	}
}

// DefaultName derives a valid node name from the hostname.
func DefaultName() string {
	h, _ := os.Hostname()
	h = strings.Split(h, ".")[0]
	clean := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.' {
			return r
		}
		return '-'
	}, h)
	clean = strings.Trim(clean, "-._")
	if len(clean) > 63 {
		clean = clean[:63]
	}
	if !namePattern.MatchString(clean) {
		return "node"
	}
	return clean
}

// Run enrols the node and writes its identity, credentials and anchors to dir.
func Run(ctx context.Context, dir *state.Dir, o Options) (state.Identity, wire.EnrollResponse, error) {
	var resp wire.EnrollResponse
	if dir.Enrolled() && !o.Force {
		return state.Identity{}, resp, errors.New("this node is already enrolled; pass --force to replace its identity")
	}
	if o.Name == "" {
		o.Name = DefaultName()
	}
	if !namePattern.MatchString(o.Name) {
		return state.Identity{}, resp, fmt.Errorf("node name %q must match %s", o.Name, namePattern)
	}
	if strings.TrimSpace(o.Token) == "" {
		return state.Identity{}, resp, errors.New("an enrolment token is required")
	}
	mode, err := o.Mode()
	if err != nil {
		return state.Identity{}, resp, err
	}
	seen := &trust.Seen{}
	httpc, base, err := mode.Client(o.Server, nil, seen, 60*time.Second)
	if err != nil {
		return state.Identity{}, resp, err
	}
	key, keyPEM, err := pki.NewKey()
	if err != nil {
		return state.Identity{}, resp, err
	}
	csr, err := pki.CSR(key)
	if err != nil {
		return state.Identity{}, resp, err
	}
	c := &controlclient.Client{HTTP: httpc, Base: base}
	resp, err = c.Enroll(ctx, wire.EnrollRequest{Token: strings.TrimSpace(o.Token), CSR: string(csr), Name: o.Name, Address: o.Address, Facts: o.Facts})
	if err != nil {
		return state.Identity{}, resp, err
	}
	if err := check(resp, key, seen.Chain(), mode); err != nil {
		return state.Identity{}, resp, err
	}
	if o.Force {
		if err := dir.Forget(); err != nil {
			return state.Identity{}, resp, err
		}
	}
	id := state.Identity{NodeID: resp.NodeID, Name: o.Name, ServerURL: base.String(), TrustDomain: resp.TrustDomain, Trust: mode.Kind,
		Address: o.Address, EnrolledAt: time.Now().UTC()}
	if mode.Kind == trust.CAPin {
		id.Fingerprint = hex.EncodeToString(mode.Pin)
	}
	for _, step := range []func() error{
		func() error { return dir.SaveWorkloadCA([]byte(resp.CA)) },
		func() error { return dir.SaveCredentials([]byte(resp.Certificate), keyPEM) },
		func() error { return dir.SavePermitKeys(resp.PermitKeys) },
		func() error { return dir.SaveIdentity(id) },
	} {
		if err := step(); err != nil {
			return state.Identity{}, resp, err
		}
	}
	return id, resp, nil
}

// check verifies the response against our key and the endpoint we actually reached.
func check(resp wire.EnrollResponse, key *ecdsa.PrivateKey, presented []*x509.Certificate, mode trust.Mode) error {
	if !ids.Valid(resp.NodeID) {
		return fmt.Errorf("starlight returned an invalid node id %q", resp.NodeID)
	}
	ca, err := pki.ParseCertPEM([]byte(resp.CA))
	if err != nil {
		return fmt.Errorf("workload CA: %w", err)
	}
	if !ca.IsCA {
		return errors.New("starlight's workload CA certificate is not a CA")
	}
	if mode.Kind == trust.CAPin {
		sum := sha256.Sum256(ca.Raw)
		if subtle.ConstantTimeCompare(sum[:], mode.Pin) != 1 {
			return errors.New("the CA starlight returned does not match --ca-sha256")
		}
	}
	pool := x509.NewCertPool()
	pool.AddCert(ca)
	if err := trust.VerifyControl(presented, pool, resp.TrustDomain); err != nil {
		return fmt.Errorf("the enrolling endpoint is not a control listener under the CA it returned: %w", err)
	}
	return verifyLeaf([]byte(resp.Certificate), ca, resp.TrustDomain, resp.NodeID, key)
}

func verifyLeaf(certPEM []byte, ca *x509.Certificate, trustDomain, nodeID string, key *ecdsa.PrivateKey) error {
	leaf, err := pki.ParseCertPEM(certPEM)
	if err != nil {
		return fmt.Errorf("node certificate: %w", err)
	}
	pub, ok := leaf.PublicKey.(*ecdsa.PublicKey)
	if !ok || !pub.Equal(&key.PublicKey) {
		return errors.New("starlight issued a certificate for a different key")
	}
	pool := x509.NewCertPool()
	pool.AddCert(ca)
	for _, usage := range []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth} {
		id, _, err := pki.Verify([]*x509.Certificate{leaf}, pool, trustDomain, usage)
		if err != nil {
			return fmt.Errorf("node certificate: %w", err)
		}
		if id.Kind != pki.KindNode || id.ID != nodeID {
			return fmt.Errorf("node certificate names %s/%s, want node/%s", id.Kind, id.ID, nodeID)
		}
	}
	return nil
}
