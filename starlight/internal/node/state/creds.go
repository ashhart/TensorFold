package state

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"sync/atomic"
)

// Creds holds the node certificate used for both mTLS directions; renewal swaps it atomically.
type Creds struct {
	cur atomic.Pointer[tls.Certificate]
}

// ParsePair decodes a combined certificate and key PEM.
func ParsePair(combined []byte) (*tls.Certificate, error) {
	pair, err := tls.X509KeyPair(combined, combined)
	if err != nil {
		return nil, err
	}
	if pair.Leaf == nil {
		leaf, err := x509.ParseCertificate(pair.Certificate[0])
		if err != nil {
			return nil, err
		}
		pair.Leaf = leaf
	}
	return &pair, nil
}

// LoadCreds reads the stored pair.
func LoadCreds(d *Dir) (*Creds, error) {
	b, err := d.CredentialsPEM()
	if err != nil {
		return nil, err
	}
	pair, err := ParsePair(b)
	if err != nil {
		return nil, err
	}
	c := &Creds{}
	c.Set(pair)
	return c, nil
}

// Get returns the current certificate for tls.Config callbacks.
func (c *Creds) Get() (*tls.Certificate, error) {
	p := c.cur.Load()
	if p == nil {
		return nil, errors.New("no node certificate loaded")
	}
	return p, nil
}

// Set replaces the current certificate.
func (c *Creds) Set(p *tls.Certificate) { c.cur.Store(p) }

// Leaf returns the parsed current certificate, or nil.
func (c *Creds) Leaf() *x509.Certificate {
	if p := c.cur.Load(); p != nil {
		return p.Leaf
	}
	return nil
}
