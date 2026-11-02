// Package guard finds credentials in request bodies before anything is sent to a cloud provider.
package guard

import (
	"bytes"
	"encoding/json"
	"io"
	"regexp"
	"sort"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
)

// Finding names one kind of credential seen in a body; values are never returned.
type Finding struct {
	Kind  string
	Count int
}

type rule struct {
	kind string
	re   *regexp.Regexp
}

var rules = []rule{
	{"private_key", regexp.MustCompile(`-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----`)},
	{"aws_access_key", regexp.MustCompile(`\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b`)},
	{"aws_secret_key", regexp.MustCompile(`(?i)aws_?secret_?(?:access_?)?key\W{0,6}[A-Za-z0-9/+]{40}\b`)},
	{"github_token", regexp.MustCompile(`\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{60,255})\b`)},
	{"gitlab_token", regexp.MustCompile(`\bglpat-[A-Za-z0-9_-]{20,}\b`)},
	{"slack_token", regexp.MustCompile(`\bxox[abposr]-[A-Za-z0-9-]{10,}\b`)},
	{"google_api_key", regexp.MustCompile(`\bAIza[0-9A-Za-z_-]{35}\b`)},
	{"stripe_key", regexp.MustCompile(`\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b`)},
	{"anthropic_key", regexp.MustCompile(`\bsk-ant-[A-Za-z0-9_-]{20,}\b`)},
	{"openai_key", regexp.MustCompile(`\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}\b`)},
	{"jwt", regexp.MustCompile(`\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{16,}\b`)},
	{"connection_string", regexp.MustCompile(`\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?)://[^\s:/@"']+:[^\s@"']{3,}@`)},
}

var starlightToken = regexp.MustCompile(`\bsl[a-z]_[a-z2-7]{16}_[a-z2-7]{47}\b`)

// ScanJSON decodes every string in a JSON body, keys included, and scans the decoded text; escapes cannot hide a key.
func ScanJSON(body []byte) []Finding {
	dec := json.NewDecoder(bytes.NewReader(body))
	var text bytes.Buffer
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			return Scan(text.Bytes())
		}
		if err != nil {
			return Scan(append(text.Bytes(), body...))
		}
		if s, ok := tok.(string); ok {
			text.WriteString(s)
			text.WriteByte('\n')
		}
	}
}

// Scan reports each credential kind found in text, at most once per kind, sorted by kind.
func Scan(body []byte) []Finding {
	counts := map[string]int{}
	for _, r := range rules {
		if n := len(r.re.FindAllIndex(body, 64)); n > 0 {
			counts[r.kind] += n
		}
	}
	for _, m := range starlightToken.FindAll(body, 64) {
		if secret.Looks(string(m)) {
			counts["starlight_token"]++
		}
	}
	out := make([]Finding, 0, len(counts))
	for k, n := range counts {
		out = append(out, Finding{Kind: k, Count: n})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Kind < out[j].Kind })
	return out
}

// Kinds lists the kinds in findings.
func Kinds(fs []Finding) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.Kind
	}
	return out
}
