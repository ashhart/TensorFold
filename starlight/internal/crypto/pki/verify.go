package pki

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"strings"
)

// Identity is a verified workload name.
type Identity struct {
	Kind Kind
	ID   string
}

// IdentityOf extracts the single SPIFFE ID of a certificate in the trust domain.
func IdentityOf(cert *x509.Certificate, trustDomain string) (Identity, error) {
	if len(cert.URIs) != 1 {
		return Identity{}, errors.New("pki: certificate must carry exactly one URI SAN")
	}
	u := cert.URIs[0]
	if u.Scheme != "spiffe" || u.Host != trustDomain || u.RawQuery != "" || u.Fragment != "" || u.User != nil {
		return Identity{}, fmt.Errorf("pki: %s is outside trust domain %s", u, trustDomain)
	}
	kind, id, ok := strings.Cut(strings.TrimPrefix(u.Path, "/"), "/")
	if !ok || strings.Contains(id, "/") || !validName(id) {
		return Identity{}, fmt.Errorf("pki: malformed workload path %q", u.Path)
	}
	switch Kind(kind) {
	case KindNode, KindGateway, KindControl:
	default:
		return Identity{}, fmt.Errorf("pki: unknown workload kind %q", kind)
	}
	return Identity{Kind: Kind(kind), ID: id}, nil
}

// Verify checks a presented chain against roots and returns the leaf's identity.
func Verify(chain []*x509.Certificate, roots *x509.CertPool, trustDomain string, usage x509.ExtKeyUsage) (Identity, *x509.Certificate, error) {
	if len(chain) == 0 {
		return Identity{}, nil, errors.New("pki: no peer certificate")
	}
	inter := x509.NewCertPool()
	for _, c := range chain[1:] {
		inter.AddCert(c)
	}
	if _, err := chain[0].Verify(x509.VerifyOptions{Roots: roots, Intermediates: inter, KeyUsages: []x509.ExtKeyUsage{usage}}); err != nil {
		return Identity{}, nil, fmt.Errorf("pki: %w", err)
	}
	id, err := IdentityOf(chain[0], trustDomain)
	return id, chain[0], err
}

// ServerConfig makes a TLS server that requires a client certificate of one kind from the CA.
func ServerConfig(leaf tls.Certificate, roots *x509.CertPool, trustDomain string, want Kind, optional bool) *tls.Config {
	auth := tls.RequireAnyClientCert
	if optional {
		auth = tls.RequestClientCert
	}
	return &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{leaf},
		ClientAuth:   auth,
		VerifyConnection: func(cs tls.ConnectionState) error {
			if len(cs.PeerCertificates) == 0 && optional {
				return nil
			}
			id, _, err := Verify(cs.PeerCertificates, roots, trustDomain, x509.ExtKeyUsageClientAuth)
			if err != nil {
				return err
			}
			if id.Kind != want {
				return fmt.Errorf("pki: peer is a %s, want %s", id.Kind, want)
			}
			return nil
		},
	}
}

// ClientConfig makes a TLS client that presents leaf and accepts only the named peer.
func ClientConfig(leaf func() (*tls.Certificate, error), roots *x509.CertPool, trustDomain string, want Identity) *tls.Config {
	return &tls.Config{
		MinVersion:         tls.VersionTLS13,
		InsecureSkipVerify: true,
		GetClientCertificate: func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
			return leaf()
		},
		VerifyConnection: func(cs tls.ConnectionState) error {
			id, _, err := Verify(cs.PeerCertificates, roots, trustDomain, x509.ExtKeyUsageServerAuth)
			if err != nil {
				return err
			}
			if id != want {
				return fmt.Errorf("pki: connected to %s/%s, want %s/%s", id.Kind, id.ID, want.Kind, want.ID)
			}
			return nil
		},
	}
}

// NewKey returns a P-256 key and its PEM encoding.
func NewKey() (*ecdsa.PrivateKey, []byte, error) {
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	der, err := x509.MarshalECPrivateKey(k)
	if err != nil {
		return nil, nil, err
	}
	return k, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: der}), nil
}

// CSR builds a certificate request carrying only the public key.
func CSR(key *ecdsa.PrivateKey) ([]byte, error) {
	der, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{}, key)
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE REQUEST", Bytes: der}), nil
}

// ParseCertPEM decodes the first certificate in data.
func ParseCertPEM(data []byte) (*x509.Certificate, error) {
	block, _ := pem.Decode(data)
	if block == nil || block.Type != "CERTIFICATE" {
		return nil, errors.New("pki: not a PEM certificate")
	}
	return x509.ParseCertificate(block.Bytes)
}
