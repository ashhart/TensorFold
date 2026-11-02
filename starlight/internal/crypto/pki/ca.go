// Package pki runs Starlight's internal CA, which names every node and gateway with a SPIFFE ID.
package pki

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net/url"
	"strings"
	"time"
)

// CA signs short-lived workload certificates inside one trust domain.
type CA struct {
	Cert        *x509.Certificate
	CertPEM     []byte
	key         crypto.Signer
	TrustDomain string
}

// NewCA creates a P-256 root constrained to URIs in the trust domain.
func NewCA(trustDomain string, validity time.Duration) (*CA, []byte, error) {
	if err := validTrustDomain(trustDomain); err != nil {
		return nil, nil, err
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	now := time.Now().Add(-time.Minute)
	tmpl := &x509.Certificate{
		SerialNumber:                serial(),
		Subject:                     pkix.Name{CommonName: "TensorFold Starlight workload CA", Organization: []string{trustDomain}},
		NotBefore:                   now,
		NotAfter:                    now.Add(validity),
		KeyUsage:                    x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid:       true,
		IsCA:                        true,
		MaxPathLenZero:              true,
		PermittedDNSDomainsCritical: true,
		PermittedURIDomains:         []string{trustDomain},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return nil, nil, err
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		return nil, nil, err
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return nil, nil, err
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	return &CA{Cert: cert, CertPEM: certPEM, key: key, TrustDomain: trustDomain}, keyDER, nil
}

// LoadCA rebuilds a CA from its certificate and PKCS#8 key.
func LoadCA(certPEM, keyDER []byte) (*CA, error) {
	block, _ := pem.Decode(certPEM)
	if block == nil || block.Type != "CERTIFICATE" {
		return nil, errors.New("pki: CA certificate is not PEM")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, err
	}
	k, err := x509.ParsePKCS8PrivateKey(keyDER)
	if err != nil {
		return nil, err
	}
	signer, ok := k.(crypto.Signer)
	if !ok {
		return nil, errors.New("pki: CA key cannot sign")
	}
	if len(cert.PermittedURIDomains) != 1 {
		return nil, errors.New("pki: CA is not constrained to one trust domain")
	}
	return &CA{Cert: cert, CertPEM: certPEM, key: signer, TrustDomain: cert.PermittedURIDomains[0]}, nil
}

// Kind is the workload role encoded in a SPIFFE path.
type Kind string

const (
	KindNode    Kind = "node"
	KindGateway Kind = "gateway"
	KindControl Kind = "control"
)

// ID formats spiffe://domain/kind/id.
func (ca *CA) ID(kind Kind, id string) *url.URL {
	return &url.URL{Scheme: "spiffe", Host: ca.TrustDomain, Path: "/" + string(kind) + "/" + id}
}

// Issue signs a leaf for pub with the role's key usages, ignoring any names a requester asked for.
func (ca *CA) Issue(kind Kind, id string, pub crypto.PublicKey, validity time.Duration) (certPEM []byte, cert *x509.Certificate, err error) {
	if !validName(id) {
		return nil, nil, fmt.Errorf("pki: invalid workload id %q", id)
	}
	switch pub.(type) {
	case *ecdsa.PublicKey:
	default:
		return nil, nil, errors.New("pki: workload keys must be ECDSA")
	}
	usage := []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	if kind == KindNode || kind == KindControl {
		usage = append(usage, x509.ExtKeyUsageServerAuth)
	}
	now := time.Now().Add(-time.Minute)
	notAfter := now.Add(validity)
	if notAfter.After(ca.Cert.NotAfter) {
		notAfter = ca.Cert.NotAfter
	}
	tmpl := &x509.Certificate{
		SerialNumber: serial(),
		Subject:      pkix.Name{CommonName: string(kind) + "/" + id},
		NotBefore:    now,
		NotAfter:     notAfter,
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  usage,
		URIs:         []*url.URL{ca.ID(kind, id)},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca.Cert, pub, ca.key)
	if err != nil {
		return nil, nil, err
	}
	cert, err = x509.ParseCertificate(der)
	if err != nil {
		return nil, nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), cert, nil
}

// SignCSR verifies a CSR's proof of possession and issues a leaf for its key.
func (ca *CA) SignCSR(kind Kind, id string, csrPEM []byte, validity time.Duration) ([]byte, *x509.Certificate, error) {
	block, _ := pem.Decode(csrPEM)
	if block == nil || block.Type != "CERTIFICATE REQUEST" {
		return nil, nil, errors.New("pki: CSR is not PEM")
	}
	csr, err := x509.ParseCertificateRequest(block.Bytes)
	if err != nil {
		return nil, nil, err
	}
	if err := csr.CheckSignature(); err != nil {
		return nil, nil, fmt.Errorf("pki: CSR signature: %w", err)
	}
	return ca.Issue(kind, id, csr.PublicKey, validity)
}

// Pool returns a pool holding only this CA.
func (ca *CA) Pool() *x509.CertPool {
	p := x509.NewCertPool()
	p.AddCert(ca.Cert)
	return p
}

func serial() *big.Int {
	n, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 127))
	if err != nil {
		panic(err)
	}
	return n.Add(n, big.NewInt(1))
}

func validTrustDomain(td string) error {
	if td == "" || len(td) > 253 || strings.ToLower(td) != td {
		return fmt.Errorf("pki: invalid trust domain %q", td)
	}
	for _, label := range strings.Split(td, ".") {
		if !validName(label) {
			return fmt.Errorf("pki: invalid trust domain %q", td)
		}
	}
	return nil
}

func validName(s string) bool {
	if s == "" || len(s) > 63 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-') {
			return false
		}
	}
	return s[0] != '-' && s[len(s)-1] != '-'
}
