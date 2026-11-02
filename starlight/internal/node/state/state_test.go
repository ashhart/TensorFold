package state

import (
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

func TestOpenTightensPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX modes")
	}
	root := filepath.Join(t.TempDir(), "s")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	d, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	fi, _ := os.Stat(d.Root)
	if fi.Mode().Perm() != 0o700 {
		t.Fatalf("mode %v", fi.Mode().Perm())
	}
	if _, err := Open(""); err == nil {
		t.Fatal("opened an empty path")
	}
}

func TestIdentityAndKeysRoundTrip(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "s"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.LoadIdentity(); err != ErrNotEnrolled || d.Enrolled() {
		t.Fatalf("empty dir: %v", err)
	}
	id := Identity{NodeID: "0190f5a0-0000-7000-8000-000000000001", Name: "n", ServerURL: "https://starlight.example:9443",
		TrustDomain: "starlight.example", Trust: "ca-sha256", EnrolledAt: time.Now().UTC().Truncate(time.Second)}
	if err := d.SaveIdentity(id); err != nil {
		t.Fatal(err)
	}
	got, err := d.LoadIdentity()
	if err != nil || got != id {
		t.Fatalf("identity %+v %v", got, err)
	}
	fi, _ := os.Stat(d.Path("node.json"))
	if runtime.GOOS != "windows" && fi.Mode().Perm() != 0o600 {
		t.Fatalf("identity mode %v", fi.Mode().Perm())
	}
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	keys := []wire.PermitKey{{KID: "k1", PublicKey: pub}, {KID: "short", PublicKey: []byte{1}}, {KID: "", PublicKey: pub}}
	if err := d.SavePermitKeys(keys); err != nil {
		t.Fatal(err)
	}
	m, err := d.PermitKeys()
	if err != nil || len(m) != 1 || m["k1"] == nil {
		t.Fatalf("permit keys %v %v", m, err)
	}
	if err := d.Forget(); err != nil || d.Enrolled() {
		t.Fatalf("forget: %v", err)
	}
}

func TestIncompleteIdentityRefused(t *testing.T) {
	d, _ := Open(filepath.Join(t.TempDir(), "s"))
	if err := os.WriteFile(d.Path("node.json"), []byte(`{"node_id":"x"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := d.LoadIdentity(); err == nil {
		t.Fatal("loaded an incomplete identity")
	}
}

func TestWriteFileAtomicLeavesNoTemporaries(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f")
	for i := 0; i < 3; i++ {
		if err := WriteFileAtomic(p, []byte{byte(i)}, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("left %d entries", len(entries))
	}
	b, _ := os.ReadFile(p)
	if len(b) != 1 || b[0] != 2 {
		t.Fatalf("content %v", b)
	}
}
