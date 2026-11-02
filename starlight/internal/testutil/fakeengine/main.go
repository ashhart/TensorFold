package fakeengine

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Version is what `--version` prints.
const Version = "tensorfold 1.2.0-fake"

// Main runs the fake binary: `--version`, `pull REPO@REV` or `serve MODEL [flags]`; it exits the process.
func Main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if p := knob("FAKE_ENGINE_ARGV_FILE"); p != "" {
		b, _ := json.Marshal(args)
		_ = os.WriteFile(p, b, 0o600)
	}
	if len(args) == 1 && args[0] == "--version" {
		fmt.Fprintln(stdout, Version)
		return 0
	}
	if len(args) == 2 && args[0] == "pull" {
		return pull(args[1], stdout, stderr)
	}
	if len(args) >= 2 && args[0] == "serve" {
		go exitWhenOrphaned()
		return serve(args[1], args[2:], stderr)
	}
	fmt.Fprintln(stderr, "usage: fake-tensorfold --version | pull REPO@REV | serve MODEL [flags]")
	return 2
}

func pull(spec string, stdout, stderr io.Writer) int {
	if knob("FAKE_ENGINE_PULL_FAIL") != "" {
		fmt.Fprintln(stderr, spec+": the hub is unreachable")
		return 1
	}
	repo, rev, ok := strings.Cut(spec, "@")
	org, name, ok2 := strings.Cut(repo, "/")
	if !ok || !ok2 || len(rev) != 40 {
		fmt.Fprintln(stderr, "pull needs REPO@REVISION with a 40-character revision")
		return 2
	}
	hub := os.Getenv("HF_HUB_CACHE")
	if hub == "" && os.Getenv("HF_HOME") != "" {
		hub = filepath.Join(os.Getenv("HF_HOME"), "hub")
	}
	if hub == "" {
		home, _ := os.UserHomeDir()
		hub = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	root := filepath.Join(hub, "models--"+org+"--"+name)
	snap := filepath.Join(root, "snapshots", rev)
	if err := os.MkdirAll(snap, 0o755); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	_ = os.MkdirAll(filepath.Join(root, "refs"), 0o755)
	_ = os.WriteFile(filepath.Join(root, "refs", "main"), []byte(rev), 0o644)
	if err := os.WriteFile(filepath.Join(snap, "config.json"), []byte(`{"model_type":"fake"}`), 0o644); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "%s: Fake (fake)\n", repo)
	return 0
}

func serve(model string, args []string, stderr io.Writer) int {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	host := fs.String("host", "127.0.0.1", "")
	port := fs.Int("port", 8080, "")
	name := fs.String("name", "", "")
	keyFile := fs.String("api-key-file", "", "")
	for _, f := range []string{"backend", "context", "parallel", "kv-cache", "reasoning-effort", "thinking-budget", "max-tokens", "prompt-cache-gib", "drafter"} {
		fs.String(f, "", "")
	}
	for _, f := range []string{"no-update-check", "vision", "no-drafts"} {
		fs.Bool(f, false, "")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fi, err := os.Stat(model); err != nil || !fi.IsDir() {
		fmt.Fprintf(stderr, "error: model %s is not a directory\n", model)
		return 1
	}
	served := *name
	if served == "" {
		served = filepath.Base(model)
	}
	var keys []Key
	if *keyFile != "" {
		var err error
		if keys, err = ParseKeyFile(*keyFile); err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return 1
		}
	}
	if rank, ok := followerRank(); ok {
		return follow(rank, stderr)
	}
	recordRank("0")
	if err := gather(stderr); err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	if ms := envMillis("FAKE_ENGINE_STARTUP_DELAY_MS"); ms > 0 {
		time.Sleep(ms)
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(*host, strconv.Itoa(*port)))
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	tps, _ := strconv.ParseFloat(knob("FAKE_ENGINE_SYNTHETIC_TPS"), 64)
	eng := New(Options{Name: served, Keys: keys, Delay: envMillis("FAKE_ENGINE_DELAY_MS"), SyntheticTPS: tps})
	var h http.Handler = eng
	if knob("FAKE_ENGINE_HEALTH_FAIL") != "" {
		h = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/health" {
				writeErr(w, http.StatusServiceUnavailable, "warming", "not ready")
				return
			}
			eng.ServeHTTP(w, r)
		})
	}
	srv := &http.Server{Handler: h, ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = srv.Serve(ln) }()
	if knob("FAKE_ENGINE_SILENT") == "" {
		fmt.Fprintf(stderr, "[tensorfold] serving %s at http://127.0.0.1:%d/v1\n", served, ln.Addr().(*net.TCPAddr).Port)
	}
	if ms := envMillis("FAKE_ENGINE_CRASH_AFTER_MS"); ms > 0 {
		go func() {
			time.Sleep(ms)
			fmt.Fprintln(stderr, "fatal: simulated crash")
			os.Exit(3)
		}()
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, os.Interrupt)
	<-sig
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return 1
	}
	return 0
}

// knob reads a test setting, also under a TF_ prefix so a real node agent's --env allowlist passes it through.
func knob(key string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return os.Getenv("TF_" + key)
}

func envMillis(key string) time.Duration {
	n, err := strconv.Atoi(knob(key))
	if err != nil || n <= 0 {
		return 0
	}
	return time.Duration(n) * time.Millisecond
}

// exitWhenOrphaned ends a fake engine whose agent went away without stopping it, so test runs leave nothing behind.
func exitWhenOrphaned() {
	parent := os.Getppid()
	for range time.Tick(500 * time.Millisecond) {
		if os.Getppid() != parent {
			os.Exit(0)
		}
	}
}
