package keyring

import (
	"bytes"
	"crypto/hkdf"
	"crypto/sha256"
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"
)

func TestSealOpenBindsContext(t *testing.T) {
	k := ForTest()
	sealed, err := k.Seal([]byte("client-secret"), "idp:1")
	if err != nil {
		t.Fatal(err)
	}
	got, err := k.Open(sealed, "idp:1")
	if err != nil || !bytes.Equal(got, []byte("client-secret")) {
		t.Fatalf("open = %q, %v", got, err)
	}
	if _, err := k.Open(sealed, "idp:2"); err == nil {
		t.Fatal("opened a value sealed for another row")
	}
	sealed[len(sealed)-1] ^= 1
	if _, err := k.Open(sealed, "idp:1"); err == nil {
		t.Fatal("opened a tampered value")
	}
}

func TestOpenRejectsMalformed(t *testing.T) {
	k := ForTest()
	for _, v := range [][]byte{nil, {1}, {2, 0}, {1, 9, 'x'}, {1, 4, 't', 'e', 's', 't', 0}} {
		if _, err := k.Open(v, "a"); err == nil {
			t.Errorf("accepted %v", v)
		}
	}
}

func TestLoadMaster(t *testing.T) {
	key := GenerateMasterKey()
	m, err := LoadMaster(key, "")
	if err != nil {
		t.Fatal(err)
	}
	m2, _ := LoadMaster(" "+key+"\n", "")
	if m.ID != m2.ID {
		t.Fatal("whitespace changed the master id")
	}
	if _, err := LoadMaster(base64.StdEncoding.EncodeToString([]byte("short")), ""); err == nil {
		t.Fatal("accepted a short key")
	}
	if _, err := LoadMaster("", ""); err == nil {
		t.Fatal("accepted no key")
	}
}

func TestDeriveIsHKDFOfTheMasterKey(t *testing.T) {
	key := GenerateMasterKey()
	file := filepath.Join(t.TempDir(), "master.key")
	if err := os.WriteFile(file, []byte(key+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	fromValue, err := LoadMaster(key, "")
	if err != nil {
		t.Fatal(err)
	}
	fromFile, err := LoadMaster("", file)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := base64.StdEncoding.DecodeString(key)
	want, err := hkdf.Key(sha256.New, raw, nil, "rate limits", 32)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(fromValue.Derive("rate limits", 32), want) || !bytes.Equal(fromFile.Derive("rate limits", 32), want) {
		t.Fatal("Derive is not HKDF-SHA256 of the master key, alike however the key is loaded")
	}
	if !bytes.Equal(New(fromFile, fromValue).Derive("rate limits", 32), want) {
		t.Fatal("the keyring does not derive from its primary master key")
	}
	other, _ := LoadMaster(GenerateMasterKey(), "")
	if bytes.Equal(fromValue.Derive("another use", 32), want) || bytes.Equal(other.Derive("rate limits", 32), want) {
		t.Fatal("another label or master key derived the same bytes")
	}
}
