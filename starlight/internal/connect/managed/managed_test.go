package managed

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

func TestParseJSONKeepsUnknownAndIgnoresPayloadKeys(t *testing.T) {
	p, err := ParseJSON([]byte(`{"ServerURL":"https://starlight.example","BridgePort":7272,"DisableBridge":true,
		"AllowManualServer":false,"PayloadUUID":"x","Sever":"typo"}`), "test")
	if err != nil {
		t.Fatal(err)
	}
	if p.ServerURL != "https://starlight.example" || p.BridgePort != 7272 || !p.DisableBridge || p.AllowManualServer == nil || *p.AllowManualServer {
		t.Fatalf("policy %+v", p)
	}
	if len(p.Unknown) != 1 || p.Unknown[0] != "Sever" {
		t.Fatalf("unknown = %v", p.Unknown)
	}
	if !p.ServerLocked() {
		t.Fatal("managed server should be locked")
	}
	if _, err := ParseJSON([]byte(`{"BridgePort":"x"}`), "test"); err == nil {
		t.Fatal("accepted a string port")
	}
}

func TestMergePrecedence(t *testing.T) {
	allow := true
	machine := Policy{ServerURL: "https://a.example", BridgePort: 7100, Sources: []string{"machine"}}
	user := Policy{ServerURL: "https://b.example", AllowManualServer: &allow, Sources: []string{"user"}}
	got := Merge(machine, user)
	if got.ServerURL != "https://b.example" || got.BridgePort != 7100 || got.ServerLocked() || len(got.Sources) != 2 {
		t.Fatalf("merge = %+v", got)
	}
}

func TestLoadJSONFilesLaterWins(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.json"), filepath.Join(dir, "b.json")
	os.WriteFile(a, []byte(`{"ServerURL":"https://a.example","BridgePort":7300}`), 0o644)
	os.WriteFile(b, []byte(`{"ServerURL":"https://b.example"}`), 0o644)
	p, err := LoadJSONFiles(a, filepath.Join(dir, "missing.json"), b)
	if err != nil {
		t.Fatal(err)
	}
	if p.ServerURL != "https://b.example" || p.BridgePort != 7300 {
		t.Fatalf("policy %+v", p)
	}
	empty, err := LoadJSONFiles(filepath.Join(dir, "none.json"))
	if err != nil || !empty.Empty() {
		t.Fatalf("missing file = %+v, %v", empty, err)
	}
}

func TestValidate(t *testing.T) {
	if err := (Policy{BridgePort: 80}).Validate(); err == nil {
		t.Fatal("accepted a privileged port")
	}
	if err := (Policy{CACertificatePEM: "junk"}).Validate(); err == nil {
		t.Fatal("accepted a junk CA")
	}
	if err := (Policy{CACertificatePEM: string(testCA(t))}).Validate(); err != nil {
		t.Fatal(err)
	}
}

func testCA(t *testing.T) []byte {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "Example Root"}, NotBefore: time.Now(),
		NotAfter: time.Now().Add(time.Hour), IsCA: true, BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}

func TestShippedLinuxSampleParses(t *testing.T) {
	p, err := LoadJSONFiles(filepath.Join("..", "..", "..", "deploy", "mdm", "linux", "connect.json"))
	if err != nil {
		t.Fatal(err)
	}
	if p.ServerURL == "" || len(p.Unknown) != 0 || p.Validate() != nil || !p.ServerLocked() {
		t.Fatalf("sample policy %+v", p)
	}
}
