// Package engine supervises TensorFold engine processes for a node's assigned replicas.
package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
)

var servedPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`)

// BackendFlag maps Starlight's backend names to the engine's --backend values.
func BackendFlag(b string) string {
	switch b {
	case "metal":
		return "mlx"
	case "cuda":
		return "cuda"
	case "hip":
		return "rocm"
	default:
		return ""
	}
}

// Args builds the `serve` argv from typed fields only; nothing from the control plane is passed verbatim.
func Args(spec wire.Spec, model, drafter, served, keyFile string) []string {
	a := []string{"serve", model, "--host", "127.0.0.1", "--port", "0", "--name", served, "--api-key-file", keyFile, "--no-update-check"}
	if b := BackendFlag(spec.Backend); b != "" {
		a = append(a, "--backend", b)
	}
	ints := []struct {
		flag string
		v    int
	}{
		{"--context", spec.Context}, {"--parallel", spec.Parallel}, {"--thinking-budget", spec.ThinkingBudget},
		{"--max-tokens", spec.MaxTokens}, {"--prompt-cache-gib", spec.PromptCacheGiB},
	}
	for _, f := range ints {
		if f.v > 0 {
			a = append(a, f.flag, strconv.Itoa(f.v))
		}
	}
	if spec.PromptCache == "off" {
		a = append(a, "--prompt-cache-gib", "0")
	}
	if spec.KVCache != "" {
		a = append(a, "--kv-cache", spec.KVCache)
	}
	if spec.ReasoningEffort != "" {
		a = append(a, "--reasoning-effort", spec.ReasoningEffort)
	}
	if spec.Vision {
		a = append(a, "--vision")
	}
	if spec.NoDrafts {
		a = append(a, "--no-drafts")
	}
	if drafter != "" {
		a = append(a, "--drafter", drafter)
	}
	return a
}

// Check refuses assignments a node must not run.
func Check(a wire.Assignment) error {
	if !servedPattern.MatchString(a.ServedName) {
		return fmt.Errorf("served name %q is not a valid model id", a.ServedName)
	}
	if a.Slots <= 0 || a.Slots > 1024 {
		return fmt.Errorf("slots must be between 1 and 1024, got %d", a.Slots)
	}
	if a.Desired != "running" && a.Desired != "stopped" {
		return fmt.Errorf("desired state %q is neither running nor stopped", a.Desired)
	}
	if err := a.Spec.Validate(); err != nil {
		return err
	}
	return checkDevices(a)
}

// Keys are per-process engine credentials: scoped inference, read-only metrics, and full for stored Responses.
type Keys struct {
	Inference string
	Read      string
	Responses string
	File      string
}

// WriteKeys creates a fresh 0600 key file in dir in the engine's scoped format.
func WriteKeys(dir string) (Keys, error) {
	k := Keys{Inference: secret.RandomString(32), Read: secret.RandomString(32), Responses: secret.RandomString(32),
		File: filepath.Join(dir, "engine.keys")}
	body := "starlight[inference]: " + k.Inference + "\nstarlight-read[read]: " + k.Read + "\nstarlight-responses: " + k.Responses + "\n"
	_ = os.Remove(k.File)
	f, err := os.OpenFile(k.File, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return Keys{}, err
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		return Keys{}, err
	}
	return k, f.Close()
}

// servingLine matches the engine's ready line, with or without its "[tensorfold]" prefix.
var servingLine = regexp.MustCompile(`serving (\S+) at http://127\.0\.0\.1:(\d{1,5})/v1`)

// ParseServing returns the served name and port from one log line.
func ParseServing(line string) (string, int, bool) {
	m := servingLine.FindStringSubmatch(line)
	if m == nil {
		return "", 0, false
	}
	port, err := strconv.Atoi(m[2])
	if err != nil || port <= 0 || port > 65535 {
		return "", 0, false
	}
	return m[1], port, true
}

// childEnv is the minimal environment an engine gets; TENSORFOLD_API_KEY and request logs are never passed.
func childEnv(extra []string, offline bool) []string {
	keep := []string{"PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "SystemRoot", "SYSTEMROOT",
		"USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "ComSpec", "PATHEXT", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
		"NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "TENSORFOLD_MEMORY_RESERVE_GIB"}
	var env []string
	for _, k := range keep {
		if v, ok := os.LookupEnv(k); ok {
			env = append(env, k+"="+v)
		}
	}
	for _, kv := range extra {
		k, _, _ := strings.Cut(kv, "=")
		if k == "TENSORFOLD_API_KEY" || k == "TENSORFOLD_REQUEST_LOG" {
			continue
		}
		env = append(env, kv)
	}
	env = append(env, "TENSORFOLD_NO_UPDATE_CHECK=1", "TENSORFOLD_NO_LIVE=1")
	if offline {
		env = append(env, "HF_HUB_OFFLINE=1")
	}
	return env
}
