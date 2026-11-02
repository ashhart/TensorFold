package server

import (
	"crypto/tls"
	"os"
	"sync"
	"time"
)

// certFile serves the public certificate and reloads it within a minute of the files changing, with no restart.
type certFile struct {
	certPath, keyPath string
	mu                sync.Mutex
	cert              *tls.Certificate
	stamp             time.Time
	checked           time.Time
}

func loadCertFile(certPath, keyPath string) (*certFile, error) {
	c := &certFile{certPath: certPath, keyPath: keyPath}
	if err := c.reload(); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *certFile) reload() error {
	cert, err := tls.LoadX509KeyPair(c.certPath, c.keyPath)
	if err != nil {
		return err
	}
	c.cert, c.stamp, c.checked = &cert, c.modified(), time.Now()
	return nil
}

func (c *certFile) modified() time.Time {
	var newest time.Time
	for _, p := range []string{c.certPath, c.keyPath} {
		if fi, err := os.Stat(p); err == nil && fi.ModTime().After(newest) {
			newest = fi.ModTime()
		}
	}
	return newest
}

// get keeps serving the last good pair when a half-written replacement fails to parse.
func (c *certFile) get(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if time.Since(c.checked) > time.Minute {
		c.checked = time.Now()
		if m := c.modified(); !m.Equal(c.stamp) {
			_ = c.reload()
		}
	}
	return c.cert, nil
}

// notAfter is the current certificate's expiry; asking also picks up a renewed file between handshakes.
func (c *certFile) notAfter() time.Time {
	cert, _ := c.get(nil)
	if cert == nil || cert.Leaf == nil {
		return time.Time{}
	}
	return cert.Leaf.NotAfter
}
