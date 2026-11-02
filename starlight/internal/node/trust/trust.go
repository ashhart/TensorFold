// Package trust decides which Starlight control endpoints a node will talk to.
package trust

import (
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
)

// Trust kinds: the first three authenticate enrolment, workload is used for every connection afterwards.
const (
	System   = "system"
	CAFile   = "ca-file"
	CAPin    = "ca-sha256"
	Workload = "workload"
)

// Mode authenticates the control endpoint.
type Mode struct {
	Kind        string
	Pool        *x509.CertPool
	Pin         []byte
	TrustDomain string
}

// Seen records the chain the endpoint presented, so enrolment can check it against the CA Starlight returns.
type Seen struct {
	mu    sync.Mutex
	chain []*x509.Certificate
}

// Chain returns the last verified peer chain.
func (s *Seen) Chain() []*x509.Certificate {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.chain
}

func (s *Seen) set(c []*x509.Certificate) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.chain = c
	s.mu.Unlock()
}

// ParsePin accepts the hex SHA-256 of a CA certificate's DER, with or without colons or a sha256: prefix.
func ParsePin(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(strings.TrimPrefix(s, "sha256:"), "SHA256:")
	b, err := hex.DecodeString(strings.ReplaceAll(s, ":", ""))
	if err != nil || len(b) != sha256.Size {
		return nil, errors.New("--ca-sha256 must be the SHA-256 of Starlight's CA certificate, 64 hex digits")
	}
	return b, nil
}

// SPIFFEDomain returns the trust domain named by a certificate's single spiffe URI.
func SPIFFEDomain(c *x509.Certificate) (string, error) {
	if len(c.URIs) != 1 || c.URIs[0].Scheme != "spiffe" || c.URIs[0].Host == "" {
		return "", errors.New("the control endpoint certificate carries no SPIFFE identity")
	}
	return c.URIs[0].Host, nil
}

// VerifyControl checks chain against roots and requires a control identity, in trustDomain when it is known.
func VerifyControl(chain []*x509.Certificate, roots *x509.CertPool, trustDomain string) error {
	if len(chain) == 0 {
		return errors.New("the control endpoint presented no certificate")
	}
	if trustDomain == "" {
		td, err := SPIFFEDomain(chain[0])
		if err != nil {
			return err
		}
		trustDomain = td
	}
	id, _, err := pki.Verify(chain, roots, trustDomain, x509.ExtKeyUsageServerAuth)
	if err != nil {
		return err
	}
	if id.Kind != pki.KindControl {
		return fmt.Errorf("the endpoint is a %s, not a Starlight control listener", id.Kind)
	}
	return nil
}

func pinned(chain []*x509.Certificate, pin []byte) *x509.CertPool {
	for _, c := range chain {
		sum := sha256.Sum256(c.Raw)
		if c.IsCA && subtle.ConstantTimeCompare(sum[:], pin) == 1 {
			p := x509.NewCertPool()
			p.AddCert(c)
			return p
		}
	}
	return nil
}

// TLSConfig returns a client config that presents cert when asked and records the verified chain in seen.
func (m Mode) TLSConfig(serverName string, cert func() (*tls.Certificate, error), seen *Seen) (*tls.Config, error) {
	cfg := &tls.Config{MinVersion: tls.VersionTLS12, ServerName: serverName}
	if cert != nil {
		cfg.GetClientCertificate = func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
			if c, err := cert(); err == nil {
				return c, nil
			}
			return &tls.Certificate{}, nil
		}
	}
	var check func([]*x509.Certificate) error
	switch m.Kind {
	case System, "":
		cfg.VerifyConnection = func(cs tls.ConnectionState) error {
			seen.set(cs.PeerCertificates)
			return nil
		}
		return cfg, nil
	case CAFile:
		if m.Pool == nil {
			return nil, errors.New("ca-file trust needs a CA bundle")
		}
		check = func(c []*x509.Certificate) error { return VerifyControl(c, m.Pool, "") }
	case CAPin:
		if len(m.Pin) != sha256.Size {
			return nil, errors.New("ca-sha256 trust needs a 32-byte pin")
		}
		check = func(c []*x509.Certificate) error {
			pool := pinned(c, m.Pin)
			if pool == nil {
				return errors.New("no certificate in the endpoint's chain is a CA matching --ca-sha256")
			}
			return VerifyControl(c, pool, "")
		}
	case Workload:
		if m.Pool == nil || m.TrustDomain == "" {
			return nil, errors.New("workload trust needs the pinned CA and trust domain")
		}
		check = func(c []*x509.Certificate) error { return VerifyControl(c, m.Pool, m.TrustDomain) }
	default:
		return nil, fmt.Errorf("unknown trust mode %q", m.Kind)
	}
	cfg.InsecureSkipVerify = true
	cfg.VerifyConnection = func(cs tls.ConnectionState) error {
		if err := check(cs.PeerCertificates); err != nil {
			return err
		}
		seen.set(cs.PeerCertificates)
		return nil
	}
	return cfg, nil
}

// Client returns an HTTPS client for server with this trust.
func (m Mode) Client(server string, cert func() (*tls.Certificate, error), seen *Seen, timeout time.Duration) (*http.Client, *url.URL, error) {
	u, err := url.Parse(strings.TrimRight(server, "/"))
	if err != nil || u.Scheme != "https" || u.Host == "" {
		return nil, nil, fmt.Errorf("server must be an https URL, got %q", server)
	}
	cfg, err := m.TLSConfig(u.Hostname(), cert, seen)
	if err != nil {
		return nil, nil, err
	}
	tr := &http.Transport{TLSClientConfig: cfg, Proxy: http.ProxyFromEnvironment, ForceAttemptHTTP2: true, MaxIdleConns: 4,
		IdleConnTimeout: 90 * time.Second, TLSHandshakeTimeout: 15 * time.Second}
	return &http.Client{Transport: tr, Timeout: timeout}, u, nil
}
