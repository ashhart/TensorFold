// Package setup prints the configuration that points a local tool at Connect's local endpoint.
package setup

import (
	"fmt"
	"regexp"
	"runtime"
	"sort"
	"strings"
)

// Tools lists the tools setup knows how to configure.
func Tools() []string {
	out := make([]string, 0, len(renderers))
	for k := range renderers {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Options describe one tool's setup.
type Options struct {
	Tool    string
	BaseURL string
	Key     string
	Model   string
	Shell   string
}

// DefaultShell is PowerShell on Windows and a POSIX shell elsewhere.
func DefaultShell() string {
	if runtime.GOOS == "windows" {
		return "powershell"
	}
	return "posix"
}

var (
	safeValue = regexp.MustCompile(`^[A-Za-z0-9._:/@\[\]-]+$`)
	renderers = map[string]func(o Options) string{
		"claude-code": claudeCode,
		"openai":      openAI,
		"codex":       codex,
	}
)

// Render returns setup text for o.Tool; every value is checked so the text is safe to paste into a shell.
func Render(o Options) (string, error) {
	r, ok := renderers[o.Tool]
	if !ok {
		return "", fmt.Errorf("unknown tool %q; choose one of %s", o.Tool, strings.Join(Tools(), ", "))
	}
	if o.Shell == "" {
		o.Shell = DefaultShell()
	}
	if o.Shell != "posix" && o.Shell != "powershell" {
		return "", fmt.Errorf("shell must be posix or powershell")
	}
	for name, v := range map[string]string{"base URL": o.BaseURL, "key": o.Key} {
		if !safeValue.MatchString(v) {
			return "", fmt.Errorf("%s %q contains characters that are not safe in a shell", name, v)
		}
	}
	if o.Model != "" && !safeValue.MatchString(o.Model) {
		return "", fmt.Errorf("model %q contains characters that are not safe in a shell", o.Model)
	}
	o.BaseURL = strings.TrimRight(o.BaseURL, "/")
	return r(o), nil
}

func export(shell, name, value string) string {
	if shell == "powershell" {
		return fmt.Sprintf("$env:%s = \"%s\"\n", name, value)
	}
	return fmt.Sprintf("export %s=\"%s\"\n", name, value)
}

func claudeCode(o Options) string {
	out := export(o.Shell, "ANTHROPIC_BASE_URL", o.BaseURL) + export(o.Shell, "ANTHROPIC_AUTH_TOKEN", o.Key)
	if o.Model != "" {
		out += export(o.Shell, "ANTHROPIC_MODEL", o.Model)
	}
	return out
}

func openAI(o Options) string {
	return export(o.Shell, "OPENAI_BASE_URL", o.BaseURL+"/v1") + export(o.Shell, "OPENAI_API_KEY", o.Key)
}

func codex(o Options) string {
	var b strings.Builder
	b.WriteString(export(o.Shell, "TENSORFOLD_CONNECT_KEY", o.Key))
	b.WriteString("\nAdd to ~/.codex/config.toml, with the top-level keys above any [table]:\n\n")
	b.WriteString("model_provider = \"tensorfold\"\n")
	if o.Model != "" {
		fmt.Fprintf(&b, "model = \"%s\"\n", o.Model)
	}
	fmt.Fprintf(&b, "\n[model_providers.tensorfold]\nname = \"TensorFold Starlight\"\nbase_url = \"%s/v1\"\n", o.BaseURL)
	b.WriteString("env_key = \"TENSORFOLD_CONNECT_KEY\"\nwire_api = \"responses\"\n")
	return b.String()
}
