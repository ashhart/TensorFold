package guard

import (
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
)

// The fixtures are split so repository secret scanners pass them; the guard sees the joined text.
const fakeAWSKey = "AKIA" + "ABCDEFGHIJKLMNOP"

func pemHeader(kind string) string { return "-----BEGIN " + kind + " PRIVATE KEY-----" }

func TestScanFindsCredentialsInJSONBodies(t *testing.T) {
	starlight, _ := secret.Generate(secret.APIKey)
	body := `{"messages":[{"role":"user","content":"deploy with ` + fakeAWSKey + ` and\n` + pemHeader("OPENSSH") + `\nb3Bl\n` +
		`and postgres://app:hunter2secret@db.example/prod and ` + starlight + ` and ghp_` + strings.Repeat("a", 36) + `"}]}`
	got := strings.Join(Kinds(Scan([]byte(body))), ",")
	want := "aws_access_key,connection_string,github_token,private_key,starlight_token"
	if got != want {
		t.Fatalf("kinds = %s, want %s", got, want)
	}
}

func TestScanIgnoresOrdinaryText(t *testing.T) {
	clean := `{"messages":[{"role":"user","content":"Explain AKIA prefixes and how sk- keys look; the word eyJ starts base64 JSON."}]}`
	if fs := Scan([]byte(clean)); len(fs) != 0 {
		t.Fatalf("false positives: %v", fs)
	}
}

func FuzzScan(f *testing.F) {
	f.Add([]byte(`{"content":"` + pemHeader("RSA") + `"}`))
	f.Fuzz(func(t *testing.T, b []byte) {
		_ = Scan(b)
	})
}

func TestScanJSONSeesThroughEscapesAndLineStarts(t *testing.T) {
	cases := map[string]string{
		"line start": `{"content":"config:\n` + fakeAWSKey + `\nend"}`,
		"tab":        `{"content":"key\tghp_` + strings.Repeat("b", 36) + `"}`,
		"unicode":    `{"content":"\u0041` + `KIAABCDEFGHIJKLMNOP"}`,
		"in a key":   `{"` + fakeAWSKey + `":"value"}`,
	}
	for name, body := range cases {
		if fs := ScanJSON([]byte(body)); len(fs) == 0 {
			t.Errorf("%s: credential not found in %s", name, body)
		}
	}
	if fs := ScanJSON([]byte(`{"content":"no secrets here, just AKIA prose"}`)); len(fs) != 0 {
		t.Fatalf("false positive %v", fs)
	}
}
