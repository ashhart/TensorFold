package trust

import (
	"crypto/sha256"
	"crypto/x509"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
)

func TestParsePin(t *testing.T) {
	raw := strings.Repeat("ab", 32)
	colons := strings.ToUpper(strings.Join(strings.SplitAfterN(strings.Repeat("ab:", 32), ":", 32), ""))
	for _, in := range []string{raw, "sha256:" + raw, strings.TrimSuffix(colons, ":")} {
		if b, err := ParsePin(in); err != nil || len(b) != 32 {
			t.Errorf("ParsePin(%q) = %v", in, err)
		}
	}
	for _, in := range []string{"", "abcd", strings.Repeat("zz", 32)} {
		if _, err := ParsePin(in); err == nil {
			t.Errorf("accepted %q", in)
		}
	}
}

func leaf(t *testing.T, ca *pki.CA, kind pki.Kind) *x509.Certificate {
	t.Helper()
	k, _, _ := pki.NewKey()
	_, c, err := ca.Issue(kind, "x1", &k.PublicKey, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestVerifyControlRequiresControlKindUnderTheCA(t *testing.T) {
	ca, keyDER, _ := pki.NewCA("starlight.example", time.Hour)
	ca, _ = pki.LoadCA(ca.CertPEM, keyDER)
	other, otherKey, _ := pki.NewCA("starlight.example", time.Hour)
	other, _ = pki.LoadCA(other.CertPEM, otherKey)
	ctl := leaf(t, ca, pki.KindControl)
	if err := VerifyControl([]*x509.Certificate{ctl, ca.Cert}, ca.Pool(), ""); err != nil {
		t.Fatal(err)
	}
	if err := VerifyControl([]*x509.Certificate{ctl}, ca.Pool(), "elsewhere.example"); err == nil {
		t.Fatal("accepted another trust domain")
	}
	if err := VerifyControl([]*x509.Certificate{leaf(t, ca, pki.KindNode)}, ca.Pool(), ""); err == nil {
		t.Fatal("accepted a node certificate as the control endpoint")
	}
	if err := VerifyControl([]*x509.Certificate{ctl}, other.Pool(), ""); err == nil {
		t.Fatal("accepted a control certificate from another CA")
	}
	sum := sha256.Sum256(ca.Cert.Raw)
	if pinned([]*x509.Certificate{ctl, ca.Cert}, sum[:]) == nil || pinned([]*x509.Certificate{ctl}, sum[:]) != nil {
		t.Fatal("pin lookup must find the CA only when the chain carries it")
	}
}

func TestModeValidation(t *testing.T) {
	for _, m := range []Mode{{Kind: CAFile}, {Kind: CAPin, Pin: []byte{1}}, {Kind: Workload}, {Kind: "other"}} {
		if _, err := m.TLSConfig("starlight.example", nil, nil); err == nil {
			t.Errorf("accepted %+v", m)
		}
	}
	if _, _, err := (Mode{Kind: System}).Client("http://starlight.example", nil, nil, time.Second); err == nil {
		t.Fatal("accepted a plain http URL")
	}
}
