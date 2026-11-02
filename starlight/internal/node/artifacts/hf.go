package artifacts

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// HubCache mirrors the engine's rule: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub.
func HubCache(getenv func(string) string) string {
	if c := getenv("HF_HUB_CACHE"); c != "" {
		return c
	}
	if h := getenv("HF_HOME"); h != "" {
		return filepath.Join(h, "hub")
	}
	home := getenv("HOME")
	if home == "" {
		home = getenv("USERPROFILE")
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// Snapshot is the exact revision directory the engine's pull writes.
func Snapshot(hub, repo, revision string) string {
	org, name, _ := strings.Cut(repo, "/")
	return filepath.Join(hub, "models--"+org+"--"+name, "snapshots", revision)
}

func (s *Stager) hub() string {
	if s.HFHome != "" {
		return filepath.Join(s.HFHome, "hub")
	}
	return HubCache(os.Getenv)
}

// Env returns the Hugging Face variables a pull needs, matching the cache Stage reads.
func (s *Stager) Env() []string {
	var env []string
	if s.HFHome != "" {
		env = append(env, "HF_HOME="+s.HFHome)
	} else {
		for _, k := range []string{"HF_HOME", "HF_HUB_CACHE"} {
			if v, ok := os.LookupEnv(k); ok {
				env = append(env, k+"="+v)
			}
		}
	}
	if v, ok := os.LookupEnv("HF_TOKEN"); ok {
		env = append(env, "HF_TOKEN="+v)
	}
	if s.HFEndpoint != "" {
		env = append(env, "HF_ENDPOINT="+s.HFEndpoint)
	}
	return env
}

func (s *Stager) stageHF(ctx context.Context, a wire.Artifact, offline bool) (string, error) {
	snap := Snapshot(s.hub(), a.Repo, a.Revision)
	if ok, _ := hasConfig(snap); ok {
		return snap, nil
	}
	if offline && s.HFEndpoint == "" {
		return "", fmt.Errorf("%s@%s: %w", a.Repo, a.Revision, ErrOffline)
	}
	if s.Engine == "" {
		return "", fmt.Errorf("%s@%s is not cached and no engine binary is configured to pull it", a.Repo, a.Revision)
	}
	timeout := s.PullTimeout
	if timeout <= 0 {
		timeout = 6 * time.Hour
	}
	pctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	s.logger().Info("pulling artifact", "repo", a.Repo, "revision", a.Revision)
	cmd := exec.CommandContext(pctx, s.Engine, "pull", a.Repo+"@"+a.Revision)
	cmd.Env = append(baseEnv(), s.Env()...)
	cmd.Env = append(cmd.Env, "TENSORFOLD_NO_UPDATE_CHECK=1", "TENSORFOLD_NO_LIVE=1")
	var tail tailBuffer
	cmd.Stdout, cmd.Stderr = &tail, &tail
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("pull %s@%s failed: %v: %s", a.Repo, a.Revision, err, tail.String())
	}
	if ok, _ := hasConfig(snap); !ok {
		return "", fmt.Errorf("pull %s@%s finished but %s holds no config.json", a.Repo, a.Revision, snap)
	}
	return snap, nil
}

func (s *Stager) cachedHF() []string {
	hub := s.hub()
	repos, err := os.ReadDir(hub)
	if err != nil {
		return nil
	}
	var out []string
	for _, r := range repos {
		name, ok := strings.CutPrefix(r.Name(), "models--")
		if !ok || !r.IsDir() {
			continue
		}
		org, model, ok := strings.Cut(name, "--")
		if !ok {
			continue
		}
		snaps, err := os.ReadDir(filepath.Join(hub, r.Name(), "snapshots"))
		if err != nil {
			continue
		}
		for _, sn := range snaps {
			a := wire.Artifact{Kind: "hf", Repo: org + "/" + model, Revision: sn.Name()}
			if a.Validate() != nil {
				continue
			}
			if ok, _ := hasConfig(filepath.Join(hub, r.Name(), "snapshots", sn.Name())); ok {
				out = append(out, a.Key())
			}
		}
	}
	return out
}

func hasConfig(dir string) (bool, error) {
	fi, err := os.Stat(filepath.Join(dir, "config.json"))
	if err != nil {
		return false, err
	}
	return !fi.IsDir(), nil
}

// baseEnv passes only what a child needs to find its home, temp space and system libraries.
func baseEnv() []string {
	keep := []string{"PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "SystemRoot", "SYSTEMROOT",
		"USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "ComSpec", "PATHEXT", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
		"https_proxy", "http_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR"}
	var env []string
	for _, k := range keep {
		if v, ok := os.LookupEnv(k); ok {
			env = append(env, k+"="+v)
		}
	}
	return env
}

// tailBuffer keeps the last 4 KiB of a child's output for error messages.
type tailBuffer struct {
	b bytes.Buffer
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.b.Write(p)
	if t.b.Len() > 8192 {
		rest := t.b.Bytes()[t.b.Len()-4096:]
		t.b.Reset()
		t.b.Write(rest)
	}
	return len(p), nil
}

func (t *tailBuffer) String() string {
	s := t.b.String()
	if len(s) > 4096 {
		s = s[len(s)-4096:]
	}
	return strings.TrimSpace(s)
}
