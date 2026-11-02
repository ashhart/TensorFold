package pki

import (
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func newCA(t *testing.T, td string) *CA {
	t.Helper()
	ca, keyDER, err := NewCA(td, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCA(ca.CertPEM, keyDER)
	if err != nil {
		t.Fatal(err)
	}
	return loaded
}

func leaf(t *testing.T, ca *CA, kind Kind, id string) tls.Certificate {
	t.Helper()
	key, keyPEM, err := NewKey()
	if err != nil {
		t.Fatal(err)
	}
	csr, _ := CSR(key)
	certPEM, _, err := ca.SignCSR(kind, id, csr, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	pair, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	return pair
}

func TestMutualTLSAcceptsOnlyNamedPeers(t *testing.T) {
	ca := newCA(t, "starlight.example")
	node := leaf(t, ca, KindNode, "node-1")
	gw := leaf(t, ca, KindGateway, "gw-1")
	otherNode := leaf(t, ca, KindNode, "node-2")

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok") }))
	srv.TLS = ServerConfig(node, ca.Pool(), ca.TrustDomain, KindGateway, false)
	srv.StartTLS()
	defer srv.Close()

	get := func(client tls.Certificate, want Identity) error {
		cfg := ClientConfig(func() (*tls.Certificate, error) { return &client, nil }, ca.Pool(), ca.TrustDomain, want)
		c := &http.Client{Transport: &http.Transport{TLSClientConfig: cfg}, Timeout: 5 * time.Second}
		resp, err := c.Get(srv.URL)
		if err == nil {
			resp.Body.Close()
		}
		return err
	}
	if err := get(gw, Identity{KindNode, "node-1"}); err != nil {
		t.Fatalf("gateway to node-1: %v", err)
	}
	if err := get(gw, Identity{KindNode, "node-2"}); err == nil {
		t.Fatal("client accepted the wrong node identity")
	}
	if err := get(otherNode, Identity{KindNode, "node-1"}); err == nil {
		t.Fatal("server accepted a node as a gateway client")
	}
}

func TestForeignCARejected(t *testing.T) {
	ca := newCA(t, "starlight.example")
	foreign := newCA(t, "starlight.example")
	cert := leaf(t, foreign, KindGateway, "gw-1")
	x, _ := x509.ParseCertificate(cert.Certificate[0])
	if _, _, err := Verify([]*x509.Certificate{x}, ca.Pool(), ca.TrustDomain, x509.ExtKeyUsageClientAuth); err == nil {
		t.Fatal("verified a certificate from another CA")
	}
}

func TestIssueIgnoresRequestedNamesAndRejectsBadIDs(t *testing.T) {
	ca := newCA(t, "starlight.example")
	key, _, _ := NewKey()
	der, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{DNSNames: []string{"evil.example"}, IPAddresses: []net.IP{net.ParseIP("192.0.2.1")}}, key)
	if err != nil {
		t.Fatal(err)
	}
	csrPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE REQUEST", Bytes: der})
	_, cert, err := ca.SignCSR(KindNode, "n1", csrPEM, time.Hour)
	if err != nil || len(cert.DNSNames) != 0 || len(cert.IPAddresses) != 0 || len(cert.URIs) != 1 {
		t.Fatalf("issued requested names: %v %v %v", cert.DNSNames, cert.IPAddresses, err)
	}
	for _, bad := range []string{"", "UPPER", "a/b", "-x"} {
		if _, _, err := ca.Issue(KindNode, bad, &key.PublicKey, time.Hour); err == nil {
			t.Errorf("issued for id %q", bad)
		}
	}
}
