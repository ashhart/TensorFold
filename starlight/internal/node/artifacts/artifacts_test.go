package artifacts_test

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/node/artifacts"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

func TestMain(m *testing.M) {
	code := m.Run()
	fakeengine.RemoveBinary()
	os.Exit(code)
}

type entry struct {
	name string
	typ  byte
	body string
	link string
}

func archive(t *testing.T, gz bool, entries ...entry) []byte {
	t.Helper()
	var buf bytes.Buffer
	var tw *tar.Writer
	var zw *gzip.Writer
	if gz {
		zw = gzip.NewWriter(&buf)
		tw = tar.NewWriter(zw)
	} else {
		tw = tar.NewWriter(&buf)
	}
	for _, e := range entries {
		h := &tar.Header{Name: e.name, Typeflag: e.typ, Mode: 0o644, Size: int64(len(e.body)), Linkname: e.link}
		if e.typ != tar.TypeReg {
			h.Size = 0
		}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if e.typ == tar.TypeReg {
			if _, err := tw.Write([]byte(e.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	tw.Close()
	if zw != nil {
		zw.Close()
	}
	return buf.Bytes()
}

func extract(t *testing.T, data []byte, lim artifacts.Limits) (string, error) {
	t.Helper()
	p := filepath.Join(t.TempDir(), "a.tar")
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(t.TempDir(), "out")
	if err := os.MkdirAll(dest, 0o700); err != nil {
		t.Fatal(err)
	}
	return dest, artifacts.Extract(p, dest, lim)
}

var big = artifacts.Limits{MaxBytes: 1 << 20, MaxFiles: 100}

func TestExtractAcceptsPlainAndGzip(t *testing.T) {
	for _, gz := range []bool{false, true} {
		data := archive(t, gz, entry{name: "model/", typ: tar.TypeDir}, entry{name: "model/config.json", typ: tar.TypeReg, body: "{}"},
			entry{name: "./model/weights.safetensors", typ: tar.TypeReg, body: "w"})
		dest, err := extract(t, data, big)
		if err != nil {
			t.Fatalf("gzip=%v: %v", gz, err)
		}
		if b, _ := os.ReadFile(filepath.Join(dest, "model", "weights.safetensors")); string(b) != "w" {
			t.Fatal("file not extracted")
		}
	}
}

func TestExtractRefusesEscapesLinksAndBombs(t *testing.T) {
	cases := map[string][]entry{
		"parent traversal":  {{name: "../evil", typ: tar.TypeReg, body: "x"}},
		"nested traversal":  {{name: "a/../../evil", typ: tar.TypeReg, body: "x"}},
		"absolute":          {{name: "/etc/evil", typ: tar.TypeReg, body: "x"}},
		"backslash":         {{name: `..\evil`, typ: tar.TypeReg, body: "x"}},
		"symlink":           {{name: "link", typ: tar.TypeSymlink, link: "/etc"}},
		"internal symlink":  {{name: "link", typ: tar.TypeSymlink, link: "config.json"}},
		"hardlink":          {{name: "hard", typ: tar.TypeLink, link: "config.json"}},
		"device":            {{name: "dev", typ: tar.TypeChar}},
		"fifo":              {{name: "pipe", typ: tar.TypeFifo}},
		"duplicate":         {{name: "a", typ: tar.TypeReg, body: "1"}, {name: "a", typ: tar.TypeReg, body: "2"}},
		"too many files":    {{name: "1", typ: tar.TypeReg}, {name: "2", typ: tar.TypeReg}, {name: "3", typ: tar.TypeReg}},
		"too many bytes":    {{name: "big", typ: tar.TypeReg, body: strings.Repeat("x", 64)}},
		"traversal in dir":  {{name: "../d/", typ: tar.TypeDir}},
		"dot-dot component": {{name: "a/./../../b", typ: tar.TypeReg, body: "x"}},
	}
	tight := artifacts.Limits{MaxBytes: 32, MaxFiles: 2}
	for name, entries := range cases {
		if _, err := extract(t, archive(t, false, entries...), tight); err == nil {
			t.Errorf("%s: extracted", name)
		}
	}
	outside := filepath.Join(filepath.Dir(t.TempDir()), "evil")
	if _, err := os.Stat(outside); err == nil {
		t.Fatal("an entry escaped the destination")
	}
}

func shaHex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func TestURLArtifactVerifiesDigestAndCaches(t *testing.T) {
	data := archive(t, true, entry{name: "qwen/", typ: tar.TypeDir}, entry{name: "qwen/config.json", typ: tar.TypeReg, body: "{}"})
	hits := 0
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "http://insecure.example/q.tar.gz", http.StatusFound)
			return
		}
		_, _ = w.Write(data)
	}))
	defer srv.Close()
	s := &artifacts.Stager{ModelsDir: t.TempDir(), Client: srv.Client()}
	ctx := context.Background()
	a := wire.Artifact{Kind: "url", URL: srv.URL + "/q.tar.gz?sig=secret", SHA256: shaHex(data)}
	dir, err := s.Stage(ctx, a, true)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(dir) != "qwen" {
		t.Fatalf("model root %s", dir)
	}
	if _, err := s.Stage(ctx, a, true); err != nil || hits != 1 {
		t.Fatalf("second stage re-downloaded: hits=%d err=%v", hits, err)
	}
	if got := s.Cached(); len(got) != 1 || got[0] != "url:"+a.SHA256 {
		t.Fatalf("cached %v", got)
	}
	wrong := wire.Artifact{Kind: "url", URL: srv.URL + "/q.tar.gz", SHA256: strings.Repeat("0", 64)}
	if _, err := s.Stage(ctx, wrong, false); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("digest mismatch = %v", err)
	}
	redir := wire.Artifact{Kind: "url", URL: srv.URL + "/redirect", SHA256: strings.Repeat("1", 64)}
	if _, err := s.Stage(ctx, redir, false); err == nil || !strings.Contains(err.Error(), "https") {
		t.Fatalf("https downgrade = %v", err)
	}
}

func TestHFArtifactPullsExactRevisionUnlessOffline(t *testing.T) {
	bin := fakeengine.Binary(t)
	home := t.TempDir()
	s := &artifacts.Stager{Engine: bin, ModelsDir: t.TempDir(), HFHome: home}
	a := wire.Artifact{Kind: "hf", Repo: "TensorFold/Qwen3.8-27B-MLX-4bit", Revision: strings.Repeat("c", 40)}
	if _, err := s.Stage(context.Background(), a, true); !errors.Is(err, artifacts.ErrOffline) {
		t.Fatalf("offline pull = %v", err)
	}
	dir, err := s.Stage(context.Background(), a, false)
	if err != nil {
		t.Fatal(err)
	}
	want := artifacts.Snapshot(filepath.Join(home, "hub"), a.Repo, a.Revision)
	if dir != want {
		t.Fatalf("served %s, want the pinned snapshot %s", dir, want)
	}
	if got, err := s.Stage(context.Background(), a, true); err != nil || got != want {
		t.Fatalf("cached offline stage = %s %v", got, err)
	}
	found := false
	for _, k := range s.Cached() {
		found = found || k == a.Key()
	}
	if !found {
		t.Fatalf("cached %v lacks %s", s.Cached(), a.Key())
	}
	mirrored := &artifacts.Stager{Engine: bin, ModelsDir: t.TempDir(), HFHome: t.TempDir(), HFEndpoint: "https://hf-mirror.example"}
	if _, err := mirrored.Stage(context.Background(), a, true); err != nil {
		t.Fatalf("offline pull through an internal mirror: %v", err)
	}
}

func TestPathArtifact(t *testing.T) {
	s := &artifacts.Stager{}
	dir := t.TempDir()
	if _, err := s.Stage(context.Background(), wire.Artifact{Kind: "path", Path: dir}, false); err == nil {
		t.Fatal("accepted a directory without config.json")
	}
	_ = os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0o600)
	if got, err := s.Stage(context.Background(), wire.Artifact{Kind: "path", Path: dir}, true); err != nil || got != dir {
		t.Fatalf("path stage = %s %v", got, err)
	}
	if _, err := s.Stage(context.Background(), wire.Artifact{Kind: "path", Path: "relative/dir"}, false); err == nil {
		t.Fatal("accepted a relative path")
	}
}

func TestHubCacheMatchesEngineRules(t *testing.T) {
	env := map[string]string{"HOME": "/opt/user"}
	get := func(k string) string { return env[k] }
	if got := artifacts.HubCache(get); got != filepath.Join("/opt/user", ".cache", "huggingface", "hub") {
		t.Fatal(got)
	}
	env["HF_HOME"] = "/hf"
	if got := artifacts.HubCache(get); got != filepath.Join("/hf", "hub") {
		t.Fatal(got)
	}
	env["HF_HUB_CACHE"] = "/cache"
	if got := artifacts.HubCache(get); got != "/cache" {
		t.Fatal(got)
	}
}
