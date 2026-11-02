package connections

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"time"
)

// Generated certificate lifetimes: IdPs pin the SAML one, so it is long-lived; the client one is rotated at the IdP.
const (
	spValidity     = 10 * 365 * 24 * time.Hour
	clientValidity = 2 * 365 * 24 * time.Hour
)

// NewSPKeyPair generates the RSA-2048 key and self-signed certificate Starlight signs SAML requests with.
func NewSPKeyPair(commonName string) (certPEM, keyPEM string, err error) {
	return newKeyPair(commonName, spValidity, x509.KeyUsageDigitalSignature|x509.KeyUsageKeyEncipherment, nil)
}

// NewClientKeyPair makes the RSA-2048 key and self-signed certificate Starlight signs private_key_jwt assertions with.
func NewClientKeyPair(commonName string) (certPEM, keyPEM string, err error) {
	return newKeyPair(commonName, clientValidity, x509.KeyUsageDigitalSignature, []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth})
}

func newKeyPair(commonName string, validity time.Duration, usage x509.KeyUsage, ext []x509.ExtKeyUsage) (string, string, error) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return "", "", err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 120))
	if err != nil {
		return "", "", err
	}
	now := time.Now().Add(-time.Hour)
	tmpl := &x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: commonName, Organization: []string{"TensorFold Starlight"}},
		NotBefore:             now,
		NotAfter:              now.Add(validity),
		KeyUsage:              usage,
		ExtKeyUsage:           ext,
		BasicConstraintsValid: true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return "", "", err
	}
	keyDER := x509.MarshalPKCS1PrivateKey(key)
	return string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})),
		string(pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: keyDER})), nil
}

// ensureSPKeys gives a SAML connection a signing pair, keeping a stored pair unless both halves are replaced.
func ensureSPKeys(in *Input, stored Secrets) error {
	s := in.Config.SAML
	if s == nil {
		return nil
	}
	if s.SPCertificate != "" && in.Secrets.SAMLKeyPEM != "" {
		_, err := tls.X509KeyPair([]byte(s.SPCertificate), []byte(in.Secrets.SAMLKeyPEM))
		if err != nil {
			return invalid("sp_certificate", "certificate and key do not match")
		}
		return nil
	}
	if s.SPCertificate != "" && stored.SAMLKeyPEM != "" {
		in.Secrets.SAMLKeyPEM = stored.SAMLKeyPEM
		return nil
	}
	cert, key, err := NewSPKeyPair("starlight-sp-" + in.Slug)
	if err != nil {
		return err
	}
	s.SPCertificate, in.Secrets.SAMLKeyPEM = cert, key
	return nil
}
