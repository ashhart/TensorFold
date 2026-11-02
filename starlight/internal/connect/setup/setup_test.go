package setup

import (
	"strings"
	"testing"
)

const key = "slc_abcdefghijklmnop_abcdefghijklmnopqrstuvwxyz234567abcdefghabcdefg"

func TestRender(t *testing.T) {
	cases := []struct {
		o    Options
		want []string
	}{
		{Options{Tool: "claude-code", BaseURL: "http://127.0.0.1:7171/", Key: key, Model: "qwen3.8-27b", Shell: "posix"},
			[]string{`export ANTHROPIC_BASE_URL="http://127.0.0.1:7171"`, `export ANTHROPIC_AUTH_TOKEN="` + key + `"`, `ANTHROPIC_MODEL="qwen3.8-27b"`}},
		{Options{Tool: "openai", BaseURL: "http://127.0.0.1:7171", Key: key, Shell: "powershell"},
			[]string{`$env:OPENAI_BASE_URL = "http://127.0.0.1:7171/v1"`, `$env:OPENAI_API_KEY = "` + key + `"`}},
		{Options{Tool: "codex", BaseURL: "http://127.0.0.1:7171", Key: key, Model: "qwen3.8-27b", Shell: "posix"},
			[]string{`model_provider = "tensorfold"`, `[model_providers.tensorfold]`, `base_url = "http://127.0.0.1:7171/v1"`,
				`env_key = "TENSORFOLD_CONNECT_KEY"`, `wire_api = "responses"`, `model = "qwen3.8-27b"`}},
	}
	for _, c := range cases {
		got, err := Render(c.o)
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("%s output missing %q:\n%s", c.o.Tool, w, got)
			}
		}
	}
}

func TestRenderRejectsUnsafeValues(t *testing.T) {
	bad := []Options{
		{Tool: "openai", BaseURL: "http://127.0.0.1:7171", Key: key + "$(id)"},
		{Tool: "openai", BaseURL: "http://127.0.0.1:7171\"; rm -rf /", Key: key},
		{Tool: "claude-code", BaseURL: "http://127.0.0.1:7171", Key: key, Model: "m`whoami`"},
		{Tool: "cursor", BaseURL: "http://127.0.0.1:7171", Key: key},
		{Tool: "openai", BaseURL: "http://127.0.0.1:7171", Key: key, Shell: "fish"},
	}
	for _, o := range bad {
		if _, err := Render(o); err == nil {
			t.Errorf("rendered %+v", o)
		}
	}
}
