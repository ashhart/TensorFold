package connections

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"strings"
)

// ClientKey parses a private_key_jwt key, RSA of at least 2048 bits or ECDSA P-256, and its certificate when given.
func ClientKey(keyPEM, certPEM string) (crypto.Signer, *x509.Certificate, error) {
	key, err := parseClientKey(keyPEM)
	if err != nil {
		return nil, nil, err
	}
	if strings.TrimSpace(certPEM) == "" {
		return key, nil, nil
	}
	cert, err := ParseCertificate(certPEM)
	if err != nil {
		return nil, nil, err
	}
	pub, ok := key.Public().(interface{ Equal(crypto.PublicKey) bool })
	if !ok || !pub.Equal(cert.PublicKey) {
		return nil, nil, invalid("signing_certificate", "the certificate does not belong to the signing key")
	}
	return key, cert, nil
}

// ParseCertificate reads the PEM certificate a private_key_jwt connection signs with, refusing anything pasted with it.
func ParseCertificate(certPEM string) (*x509.Certificate, error) {
	return parseCertificate("signing_certificate", certPEM)
}

// CertificatePEM encodes only the certificate itself, which is all Starlight keeps or shows of a certificate field.
func CertificatePEM(cert *x509.Certificate) string {
	return string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw}))
}

// parseCertificate refuses a private key or other PEM block beside the certificate, as the field is stored unsealed.
func parseCertificate(field, certPEM string) (*x509.Certificate, error) {
	if strings.Contains(certPEM, "PRIVATE KEY") {
		return nil, invalid(field, "this holds a private key, but this field is stored unsealed; paste only the certificate here and the key in its own field")
	}
	block, rest := pem.Decode([]byte(certPEM))
	if block == nil || block.Type != "CERTIFICATE" {
		return nil, invalid(field, "paste the certificate in PEM form")
	}
	if extra, _ := pem.Decode(rest); extra != nil {
		return nil, invalid(field, "paste only the certificate, without a chain or other PEM blocks; this also holds a %s block", extra.Type)
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, invalid(field, "the certificate cannot be read: %v", err)
	}
	return cert, nil
}

func parseClientKey(keyPEM string) (crypto.Signer, error) {
	block, _ := pem.Decode([]byte(keyPEM))
	if block == nil {
		return nil, invalid("signing_key_pem", "paste the private key in PEM form")
	}
	var key any
	if k, err := x509.ParsePKCS8PrivateKey(block.Bytes); err == nil {
		key = k
	} else if k, err := x509.ParsePKCS1PrivateKey(block.Bytes); err == nil {
		key = k
	} else if k, err := x509.ParseECPrivateKey(block.Bytes); err == nil {
		key = k
	} else {
		return nil, invalid("signing_key_pem", "use an unencrypted PKCS #8, PKCS #1 or SEC 1 key")
	}
	switch k := key.(type) {
	case *rsa.PrivateKey:
		if k.N.BitLen() < 2048 {
			return nil, invalid("signing_key_pem", "RSA keys must have at least 2048 bits")
		}
		return k, nil
	case *ecdsa.PrivateKey:
		if k.Curve != elliptic.P256() {
			return nil, invalid("signing_key_pem", "EC keys must use the P-256 curve, for ES256")
		}
		return k, nil
	}
	return nil, invalid("signing_key_pem", "the key must be RSA or EC")
}

// checkClientKey requires a usable key, and a matching certificate when one is set, after stored secrets are merged in.
func checkClientKey(in Input) error {
	o := in.Config.OIDC
	if o == nil || (o.AuthMethod != AuthPrivateJWT && o.SigningCertificate == "") {
		return nil
	}
	if strings.TrimSpace(in.Secrets.SigningKeyPEM) == "" {
		return invalid("signing_key_pem", "certificate authentication needs a private key; paste it with its certificate, or generate a pair")
	}
	_, cert, err := ClientKey(in.Secrets.SigningKeyPEM, o.SigningCertificate)
	if err != nil {
		return err
	}
	if cert == nil && in.Preset == "entra" && o.AuthMethod == AuthPrivateJWT {
		return invalid("signing_certificate", "Microsoft Entra ID finds the key by its certificate; paste the certificate or generate a pair")
	}
	return nil
}

// RotateClientKey replaces an OIDC connection's private_key_jwt key and certificate with a newly generated pair.
func (s *Store) RotateClientKey(ctx context.Context, id string, rec Recorder) (Connection, error) {
	cur, err := s.Get(ctx, id)
	if err != nil {
		return Connection{}, err
	}
	if cur.Protocol != OIDC || cur.Config.OIDC == nil {
		return Connection{}, invalid("signing_certificate", "only OpenID Connect connections use a client certificate")
	}
	cert, key, err := NewClientKeyPair("starlight-" + cur.Slug)
	if err != nil {
		return Connection{}, err
	}
	o := *cur.Config.OIDC
	o.SigningCertificate = cert
	in := Input{Slug: cur.Slug, Name: cur.Name, Protocol: cur.Protocol, Preset: cur.Preset, Domains: cur.Domains, Enabled: cur.Enabled, JIT: cur.JIT,
		Config: cur.Config, Secrets: Secrets{SigningKeyPEM: key}}
	in.Config.OIDC = &o
	return s.Update(ctx, id, in, rec)
}
