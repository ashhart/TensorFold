package nodetest

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
)

// ModelDir creates a directory that passes path-artifact staging.
func ModelDir(t testing.TB) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "model")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"fake"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

// Assignment builds a runnable assignment serving a local model directory.
func Assignment(replicaID, served, modelDir string, slots int, gen int64) wire.Assignment {
	return wire.Assignment{ReplicaID: replicaID, DeploymentID: "dep-1", Revision: 1, Generation: gen, ServedName: served,
		Desired: "running", Slots: slots, Spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: modelDir}}}
}

// Permit signs claims for a request body with sensible defaults filled in.
func Permit(t testing.TB, s permit.Signer, c permit.Claims, body []byte) string {
	t.Helper()
	sum := sha256.Sum256(body)
	if c.BodySHA256 == "" {
		c.BodySHA256 = hex.EncodeToString(sum[:])
	}
	now := time.Now()
	if c.IssuedAt == 0 {
		c.IssuedAt = now.UnixMilli()
	}
	if c.Expires == 0 {
		c.Expires = now.Add(30 * time.Second).UnixMilli()
	}
	tok, err := s.Sign(c)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}
