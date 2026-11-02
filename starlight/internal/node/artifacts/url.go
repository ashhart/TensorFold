package artifacts

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
)

const (
	marker          = ".starlight-artifact"
	defaultMaxBytes = 4 << 40
	defaultMaxFiles = 200000
)

func (s *Stager) urlRoot() string { return filepath.Join(s.ModelsDir, "url") }

func (s *Stager) stageURL(ctx context.Context, a wire.Artifact) (string, error) {
	if s.ModelsDir == "" {
		return "", errors.New("no models directory configured for URL artifacts")
	}
	final := filepath.Join(s.urlRoot(), a.SHA256)
	if _, err := os.Stat(filepath.Join(final, marker)); err == nil {
		return modelRoot(final)
	}
	for _, d := range []string{s.ModelsDir, s.urlRoot(), filepath.Join(s.ModelsDir, "downloads")} {
		if err := state.EnsurePrivateDir(d); err != nil {
			return "", err
		}
	}
	part := filepath.Join(s.ModelsDir, "downloads", a.SHA256+".part")
	defer os.Remove(part)
	s.logger().Info("downloading artifact", "url", redact(a.URL), "sha256", a.SHA256)
	if err := s.download(ctx, a.URL, part, a.SHA256); err != nil {
		return "", err
	}
	tmp, err := os.MkdirTemp(s.urlRoot(), "."+a.SHA256+".extract-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmp)
	if err := Extract(part, tmp, s.limits()); err != nil {
		return "", err
	}
	if _, err := modelRoot(tmp); err != nil {
		return "", err
	}
	meta, _ := json.Marshal(map[string]string{"sha256": a.SHA256, "url": redact(a.URL), "staged_at": time.Now().UTC().Format(time.RFC3339)})
	if err := os.WriteFile(filepath.Join(tmp, marker), meta, 0o600); err != nil {
		return "", err
	}
	if err := os.RemoveAll(final); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, final); err != nil {
		return "", err
	}
	return modelRoot(final)
}

func (s *Stager) limits() Limits {
	l := Limits{MaxBytes: s.MaxExtractBytes, MaxFiles: s.MaxFiles}
	if l.MaxBytes <= 0 {
		l.MaxBytes = defaultMaxBytes
	}
	if l.MaxFiles <= 0 {
		l.MaxFiles = defaultMaxFiles
	}
	return l
}

func (s *Stager) download(ctx context.Context, rawURL, dest, wantSHA string) error {
	client := s.Client
	if client == nil {
		client = &http.Client{Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, ResponseHeaderTimeout: time.Minute}}
	}
	c := *client
	c.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if req.URL.Scheme != "https" {
			return errors.New("artifact download redirected away from https")
		}
		if len(via) >= 5 {
			return errors.New("artifact download redirected too many times")
		}
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	resp, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", redact(rawURL), err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", redact(rawURL), resp.Status)
	}
	max := s.MaxArchiveBytes
	if max <= 0 {
		max = defaultMaxBytes
	}
	if resp.ContentLength > max {
		return fmt.Errorf("artifact is %d bytes, above the %d byte cap", resp.ContentLength, max)
	}
	f, err := os.OpenFile(dest, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(f, h), io.LimitReader(resp.Body, max+1))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("download %s: %w", redact(rawURL), err)
	}
	if n > max {
		return fmt.Errorf("artifact exceeds the %d byte cap", max)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != wantSHA {
		return fmt.Errorf("artifact sha256 is %s, want %s", got, wantSHA)
	}
	return nil
}

// modelRoot finds the served directory: the root or its single top-level folder, holding config.json.
func modelRoot(dir string) (string, error) {
	if ok, _ := hasConfig(dir); ok {
		return dir, nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var dirs []string
	for _, e := range entries {
		if e.Name() == marker {
			continue
		}
		if !e.IsDir() {
			return "", fmt.Errorf("artifact has no config.json at its root")
		}
		dirs = append(dirs, e.Name())
	}
	if len(dirs) == 1 {
		inner := filepath.Join(dir, dirs[0])
		if ok, _ := hasConfig(inner); ok {
			return inner, nil
		}
	}
	return "", fmt.Errorf("artifact has no config.json at its root or in a single top-level folder")
}

func (s *Stager) cachedURL() []string {
	entries, err := os.ReadDir(s.urlRoot())
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		if _, err := os.Stat(filepath.Join(s.urlRoot(), e.Name(), marker)); err == nil {
			out = append(out, "url:"+e.Name())
		}
	}
	return out
}

// redact drops query strings, which often carry signed-URL credentials, from logs.
func redact(raw string) string {
	if i := strings.IndexAny(raw, "?#"); i >= 0 {
		return raw[:i]
	}
	return raw
}
