package gateway

import (
	"errors"
	"math"
	"net/http"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

func TestParseBodyRefusesAmbiguity(t *testing.T) {
	for _, bad := range []string{
		`{"model":"a","model":"b"}`, `[]`, `{"model":"a"} {"x":1}`, `{"model":`, `"x"`, ``, `{"a":1,}`,
	} {
		if _, err := ParseBody([]byte(bad)); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
}

func TestBodyRoundTripAndRewrite(t *testing.T) {
	b, err := ParseBody([]byte(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":64}`))
	if err != nil {
		t.Fatal(err)
	}
	_ = b.Set("model", "qwen3.8-27b")
	_ = b.Set("stream_options", map[string]bool{"include_usage": true})
	b.Delete("stream")
	got := string(b.Bytes())
	want := `{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"stream_options":{"include_usage":true}}`
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}

func TestInspect(t *testing.T) {
	raw := `{"model":"m","stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":100,"max_tokens":5,` +
		`"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("A", 90000) + `"}}]}]}`
	b, err := ParseBody([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	in, err := inspect(providers.ChatCompletions, b, []byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if !in.Stream || !in.WantsUsage || in.MaxTokens != 100 || in.MaxField != "max_completion_tokens" || in.Estimate > 2000 || in.Affinity == 0 {
		t.Fatalf("info = %+v", in)
	}
	b, _ = ParseBody([]byte(`{"model":"m","input":"hi","store":false,"previous_response_id":"resp_1"}`))
	in, _ = inspect(providers.Responses, b, nil)
	if in.Store || in.PreviousID != "resp_1" || in.MaxField != "max_output_tokens" {
		t.Fatalf("responses info = %+v", in)
	}
	for _, bad := range []string{`{"messages":[]}`, `{"model":"m","max_tokens":-1}`, `{"model":"m","stream":"yes"}`, `{"model":"m","max_tokens":1.5}`} {
		b, _ := ParseBody([]byte(bad))
		if _, err := inspect(providers.ChatCompletions, b, []byte(bad)); err == nil {
			t.Errorf("accepted %s", bad)
		}
	}
}

func TestInspectBoundsReservations(t *testing.T) {
	for protocol, bodies := range map[string][]string{
		providers.ChatCompletions: {`{"model":"m","max_tokens":9223372036854775807}`, `{"model":"m","max_completion_tokens":1048577}`,
			`{"model":"m","n":17}`, `{"model":"m","best_of":4294967296}`},
		providers.Responses: {`{"model":"m","max_output_tokens":2000000}`},
		providers.Messages:  {`{"model":"m","max_tokens":9223372036854775807}`},
	} {
		for _, raw := range bodies {
			b, _ := ParseBody([]byte(raw))
			var he *httpx.Error
			if _, err := inspect(protocol, b, []byte(raw)); !errors.As(err, &he) || he.Status != http.StatusBadRequest {
				t.Errorf("%s accepted %s: %v", protocol, raw, err)
			}
		}
	}
	raw := `{"model":"m","max_tokens":1048576,"n":16}`
	b, _ := ParseBody([]byte(raw))
	if in, err := inspect(providers.ChatCompletions, b, []byte(raw)); err != nil || in.MaxTokens != maxOutput || in.Choices != maxChoices {
		t.Fatalf("at the bounds: %+v %v", in, err)
	}
}

func TestApplyMaxCannotOverflow(t *testing.T) {
	b, _ := ParseBody([]byte(`{"model":"m"}`))
	in := Info{MaxField: "max_tokens", HasMax: true, MaxTokens: math.MaxInt64, Estimate: 40}
	if err := applyMax(b, &in, catalog.Grant{MaxContext: 1000}, 4096, providers.ChatCompletions); err == nil {
		t.Fatal("an output limit near the int64 maximum passed the context check")
	}
	in = Info{MaxField: "max_tokens"}
	if err := applyMax(b, &in, catalog.Grant{}, math.MaxInt, providers.ChatCompletions); err != nil || in.MaxTokens != maxOutput {
		t.Fatalf("a huge configured default became an output limit of %d: %v", in.MaxTokens, err)
	}
}

func FuzzParseBody(f *testing.F) {
	f.Add([]byte(`{"model":"m","messages":[]}`))
	f.Fuzz(func(t *testing.T, raw []byte) {
		b, err := ParseBody(raw)
		if err != nil {
			return
		}
		again, err := ParseBody(b.Bytes())
		if err != nil || string(again.Bytes()) != string(b.Bytes()) {
			t.Fatalf("re-encoding is not stable: %q", raw)
		}
	})
}
