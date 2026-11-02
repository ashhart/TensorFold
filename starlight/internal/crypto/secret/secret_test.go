package secret

import (
	"bytes"
	"strings"
	"testing"
)

func TestGenerateParses(t *testing.T) {
	for k := range known {
		tok, h := Generate(k)
		got, ok := Parse(tok)
		if !ok || got != k {
			t.Fatalf("%s: parse(%q) = %q, %v", k, tok, got, ok)
		}
		if !bytes.Equal(h, Hash(tok)) {
			t.Fatal("hash mismatch")
		}
		if !strings.HasPrefix(PublicID(tok), string(k)+"_") || len(PublicID(tok)) != 20 {
			t.Fatalf("public id %q", PublicID(tok))
		}
	}
}

func TestParseRejectsTampering(t *testing.T) {
	tok, _ := Generate(APIKey)
	flip := []byte(tok)
	if flip[10] == 'a' {
		flip[10] = 'b'
	} else {
		flip[10] = 'a'
	}
	for _, bad := range []string{"", tok[:len(tok)-1], string(flip), "xyz" + tok[3:], strings.ToUpper(tok), tok + "a"} {
		if _, ok := Parse(bad); ok {
			t.Errorf("accepted %q", bad)
		}
	}
}

func TestUserCode(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 1000; i++ {
		c := UserCode()
		if len(c) != 9 || c[4] != '-' {
			t.Fatalf("bad code %q", c)
		}
		if NormalizeUserCode(strings.ToLower(strings.ReplaceAll(c, "-", " "))) != c {
			t.Fatalf("normalize failed for %q", c)
		}
		seen[c] = true
	}
	if len(seen) < 990 {
		t.Fatalf("too many collisions: %d unique", len(seen))
	}
	for _, bad := range []string{"", "ABCD-EFG", "AAAA-AAAA", "BCDF-GHJ1"} {
		if NormalizeUserCode(bad) != "" {
			t.Errorf("accepted %q", bad)
		}
	}
}
