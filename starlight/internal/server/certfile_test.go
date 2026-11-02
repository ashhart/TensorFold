package server

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writePair(t *testing.T, dir string, notAfter time.Time) {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(time.Now().UnixNano()), Subject: pkix.Name{CommonName: "starlight.example"},
		DNSNames: []string{"starlight.example"}, NotBefore: time.Now().Add(-time.Hour), NotAfter: notAfter}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	kder, _ := x509.MarshalPKCS8PrivateKey(key)
	_ = os.WriteFile(filepath.Join(dir, "tls.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600)
	_ = os.WriteFile(filepath.Join(dir, "tls.key"), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: kder}), 0o600)
}

func TestPublicCertificateReloadsWithoutRestart(t *testing.T) {
	dir := t.TempDir()
	first := time.Now().Add(30 * 24 * time.Hour).Truncate(time.Second)
	writePair(t, dir, first)
	c, err := loadCertFile(filepath.Join(dir, "tls.crt"), filepath.Join(dir, "tls.key"))
	if err != nil {
		t.Fatal(err)
	}
	if !c.notAfter().Equal(first) {
		t.Fatalf("expiry %s, want %s", c.notAfter(), first)
	}
	renewed := time.Now().Add(90 * 24 * time.Hour).Truncate(time.Second)
	writePair(t, dir, renewed)
	later := time.Now().Add(time.Minute)
	_ = os.Chtimes(filepath.Join(dir, "tls.crt"), later, later)
	if err := os.WriteFile(filepath.Join(dir, "tls.key"), []byte("half written"), 0o600); err != nil {
		t.Fatal(err)
	}
	c.checked = time.Time{}
	if !c.notAfter().Equal(first) {
		t.Fatal("a broken replacement displaced the working certificate")
	}
	writePair(t, dir, renewed)
	_ = os.Chtimes(filepath.Join(dir, "tls.key"), later.Add(time.Second), later.Add(time.Second))
	c.checked = time.Time{}
	if !c.notAfter().Equal(renewed) {
		t.Fatalf("renewed certificate not picked up: %s", c.notAfter())
	}
}
