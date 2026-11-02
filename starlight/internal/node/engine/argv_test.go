package engine

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

func TestArgsFromTypedSpecOnly(t *testing.T) {
	spec := wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, Backend: "metal", Context: 32768, Parallel: 8, KVCache: "fp8",
		ReasoningEffort: "high", ThinkingBudget: 2048, MaxTokens: 4096, PromptCacheGiB: 16, Vision: true, NoDrafts: true}
	got := Args(spec, "/m", "/d", "qwen3.8-27b", "/s/engine.keys")
	want := []string{"serve", "/m", "--host", "127.0.0.1", "--port", "0", "--name", "qwen3.8-27b", "--api-key-file", "/s/engine.keys",
		"--no-update-check", "--backend", "mlx", "--context", "32768", "--parallel", "8", "--thinking-budget", "2048", "--max-tokens", "4096",
		"--prompt-cache-gib", "16", "--kv-cache", "fp8", "--reasoning-effort", "high", "--vision", "--no-drafts", "--drafter", "/d"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args\n got %q\nwant %q", got, want)
	}
	minimal := Args(wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}}, "/m", "", "m", "/k")
	if len(minimal) != 11 {
		t.Fatalf("zero-valued fields must add no flags: %q", minimal)
	}
}

func TestBackendFlag(t *testing.T) {
	for in, want := range map[string]string{"metal": "mlx", "cuda": "cuda", "hip": "rocm", "xpu": "", "": ""} {
		if got := BackendFlag(in); got != want {
			t.Errorf("BackendFlag(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCheckRefusesUnsafeAssignments(t *testing.T) {
	ok := wire.Assignment{ServedName: "qwen3.8-27b", Slots: 4, Desired: "running", Spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}}}
	if err := Check(ok); err != nil {
		t.Fatal(err)
	}
	bad := []func(*wire.Assignment){
		func(a *wire.Assignment) { a.ServedName = "--api-key=x" },
		func(a *wire.Assignment) { a.ServedName = "" },
		func(a *wire.Assignment) { a.Slots = 0 },
		func(a *wire.Assignment) { a.Desired = "maybe" },
		func(a *wire.Assignment) { a.Spec.Model = wire.Artifact{Kind: "path", Path: "relative"} },
		func(a *wire.Assignment) { a.Spec.KVCache = "int4" },
	}
	for i, mut := range bad {
		a := ok
		mut(&a)
		if Check(a) == nil {
			t.Errorf("case %d accepted %+v", i, a)
		}
	}
}

func TestParseServing(t *testing.T) {
	cases := map[string]int{
		"[tensorfold] serving qwen at http://127.0.0.1:51234/v1": 51234,
		"serving qwen at http://127.0.0.1:8080/v1":               8080,
	}
	for line, port := range cases {
		name, got, ok := ParseServing(line)
		if !ok || name != "qwen" || got != port {
			t.Errorf("ParseServing(%q) = %q %d %v", line, name, got, ok)
		}
	}
	for _, line := range []string{"serving qwen at http://0.0.0.0:8080/v1", "serving qwen at http://127.0.0.1:99999/v1", "loading qwen"} {
		if _, _, ok := ParseServing(line); ok {
			t.Errorf("accepted %q", line)
		}
	}
}

func TestWriteKeysIsPrivateAndScoped(t *testing.T) {
	dir := t.TempDir()
	k, err := WriteKeys(dir)
	if err != nil {
		t.Fatal(err)
	}
	fi, _ := os.Stat(k.File)
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("key file mode %v", fi.Mode().Perm())
	}
	b, _ := os.ReadFile(k.File)
	if !strings.Contains(string(b), "starlight[inference]: "+k.Inference) || !strings.Contains(string(b), "starlight-read[read]: "+k.Read) {
		t.Fatalf("key file %q", b)
	}
	k2, _ := WriteKeys(dir)
	if k2.Inference == k.Inference {
		t.Fatal("keys are not fresh per process")
	}
}

func TestChildEnvNeverCarriesKeys(t *testing.T) {
	t.Setenv("TENSORFOLD_API_KEY", "leak")
	env := childEnv([]string{"TENSORFOLD_API_KEY=x", "TENSORFOLD_REQUEST_LOG=/tmp/x", "TENSORFOLD_MEMORY_RESERVE_GIB=8"}, true)
	joined := strings.Join(env, "\n")
	if strings.Contains(joined, "TENSORFOLD_API_KEY") || strings.Contains(joined, "REQUEST_LOG") {
		t.Fatalf("leaked: %s", joined)
	}
	for _, want := range []string{"TENSORFOLD_NO_UPDATE_CHECK=1", "TENSORFOLD_NO_LIVE=1", "HF_HUB_OFFLINE=1", "TENSORFOLD_MEMORY_RESERVE_GIB=8"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %s", want)
		}
	}
	if strings.Contains(strings.Join(childEnv(nil, false), "\n"), "HF_HUB_OFFLINE") {
		t.Error("HF_HUB_OFFLINE set while online")
	}
}

func TestParseSample(t *testing.T) {
	text := "# TYPE tensorfold:requests_waiting gauge\ntensorfold:requests_waiting{pool=\"a\"} 2\ntensorfold:requests_waiting{pool=\"b\"} 3\n" +
		"tensorfold:requests_waiting_total 99\ntensorfold:kv_cache_usage_ratio{pool=\"0\"} 0.25\ntensorfold:kv_cache_usage_ratio{pool=\"1\"} 0.75\n" +
		"tensorfold:generation_tokens_total 1e3\nbogus{ 1\n"
	s := &sample{fam: parseSample(strings.NewReader(text))}
	if n, ok := s.sum(famWaiting); !ok || n != 5 {
		t.Fatalf("waiting = %v %v", n, ok)
	}
	if kv, ok := s.max(famKV); !ok || kv != 0.75 {
		t.Fatalf("kv = %v %v", kv, ok)
	}
	if g, ok := produced(s); !ok || g != 1000 {
		t.Fatalf("produced = %v %v", g, ok)
	}
	if _, ok := s.sum(famRunning); ok {
		t.Fatal("found an absent family")
	}
}

func TestDefaultBackoff(t *testing.T) {
	if DefaultBackoff(1).Seconds() != 1 || DefaultBackoff(3).Seconds() != 4 || DefaultBackoff(40).Seconds() != 60 {
		t.Fatalf("backoff %v %v %v", DefaultBackoff(1), DefaultBackoff(3), DefaultBackoff(40))
	}
}

func TestProbeReadsMaxBatchWhenTheEngineRevealsIt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			_, _ = io.WriteString(w, `{"status":"ok","max_batch_size":12}`)
		case "/v1/models":
			if r.Header.Get("Authorization") != "Bearer inf" {
				w.WriteHeader(401)
				return
			}
			_, _ = io.WriteString(w, `{"data":[{"id":"m"}]}`)
		}
	}))
	defer srv.Close()
	s := &Supervisor{http: srv.Client()}
	ok, batch := s.probe(context.Background(), srv.URL, "m", Keys{Inference: "inf", Read: "rd"})
	if !ok || batch != 12 {
		t.Fatalf("probe = %v %d", ok, batch)
	}
	if ok, _ := s.probe(context.Background(), srv.URL, "other", Keys{Inference: "inf"}); ok {
		t.Fatal("ready although the served id is missing")
	}
}

func TestKeyFileCarriesAnUnscopedResponsesKey(t *testing.T) {
	k, err := WriteKeys(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	keys, err := fakeengine.ParseKeyFile(k.File)
	if err != nil || len(keys) != 3 {
		t.Fatalf("keys %+v %v", keys, err)
	}
	want := map[string]string{"starlight": "inference", "starlight-read": "read", "starlight-responses": ""}
	for _, got := range keys {
		scopes := strings.Join(got.Scopes, ",")
		if s, ok := want[got.Label]; !ok || s != scopes {
			t.Errorf("key %s has scopes %q", got.Label, scopes)
		}
	}
	if k.Responses == "" || k.Responses == k.Inference || k.Responses == k.Read {
		t.Fatal("the responses key must be distinct")
	}
}

func TestRates(t *testing.T) {
	at := time.Unix(1000, 0)
	mk := func(sec float64, base string, gen, running, prompt float64) *sample {
		return &sample{at: at.Add(time.Duration(sec * float64(time.Second))), base: base,
			fam: map[string][]float64{famGenTotal: {gen}, famGenRunning: {running}, famPrompt: {prompt}}}
	}
	a := mk(0, "b", 100, 0, 1000)
	if r := rate(nil, a, produced); r != 0 {
		t.Fatalf("first sample rate %v", r)
	}
	b := mk(5, "b", 100, 50, 1500)
	if r := rate(a, b, produced); r != 10 {
		t.Fatalf("running tokens must count while streaming: %v", r)
	}
	if r := rate(a, b, prompted); r != 100 {
		t.Fatalf("prompt rate %v", r)
	}
	c := mk(10, "b", 180, 0, 1500)
	if r := rate(b, c, produced); r != 6 {
		t.Fatalf("finishing a reply must not double count: %v", r)
	}
	if r := rate(c, mk(15, "b", 3, 0, 10), produced); r != 0 {
		t.Fatalf("counter reset rate %v", r)
	}
	if r := rate(c, mk(15, "other", 500, 0, 2000), produced); r != 0 {
		t.Fatalf("new process rate %v", r)
	}
	if r := rate(c, &sample{at: c.at.Add(time.Second), base: "b", fam: map[string][]float64{}}, produced); r != 0 {
		t.Fatalf("missing family rate %v", r)
	}
}

func TestPromptCacheOffAsksTheEngineToKeepNoStates(t *testing.T) {
	args := Args(wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, PromptCache: "off"}, "/m", "", "m", "/k")
	if i := slices.Index(args, "--prompt-cache-gib"); i < 0 || args[i+1] != "0" {
		t.Fatalf("args %v", args)
	}
	if (wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, PromptCache: "off", PromptCacheGiB: 4}).Validate() == nil {
		t.Fatal("a cache size with the cache off was accepted")
	}
}
