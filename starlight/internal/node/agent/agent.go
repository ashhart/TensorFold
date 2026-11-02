// Package agent wires a node's supervisor, permit-gated listener and control loop together.
package agent

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/node/artifacts"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/enroll"
	"github.com/ashhart/TensorFold/starlight/internal/node/facts"
	"github.com/ashhart/TensorFold/starlight/internal/node/proxy"
	"github.com/ashhart/TensorFold/starlight/internal/node/reporter"
	"github.com/ashhart/TensorFold/starlight/internal/node/service"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
	"github.com/ashhart/TensorFold/starlight/internal/node/syncer"
)

// Config is everything `starlight-node run` accepts.
type Config struct {
	StateDir       string
	Listen         string
	Advertise      string
	Engine         string
	ModelsDir      string
	HFHome         string
	HFEndpoint     string
	PreventSleep   bool
	StartupTimeout time.Duration
	DrainTimeout   time.Duration
	StatsInterval  time.Duration
	Env            []string
	RankHost       string
	Facts          func(context.Context) wire.Facts
	Logger         *slog.Logger
}

// Run serves until ctx ends, then drains replicas before closing the listener.
func Run(ctx context.Context, cfg Config) error {
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	dir, err := state.Open(cfg.StateDir)
	if err != nil {
		return err
	}
	id, err := dir.LoadIdentity()
	if err != nil {
		return err
	}
	creds, err := state.LoadCreds(dir)
	if err != nil {
		return err
	}
	pool, _, err := dir.WorkloadCA()
	if err != nil {
		return fmt.Errorf("workload CA: %w", err)
	}
	keys, err := dir.PermitKeys()
	if err != nil {
		return err
	}
	bin, err := resolveEngine(cfg.Engine)
	if err != nil {
		return err
	}
	models := cfg.ModelsDir
	if models == "" {
		if models, err = dir.Sub("models"); err != nil {
			return err
		}
	}
	replicas, err := dir.Sub("replicas")
	if err != nil {
		return err
	}
	stager := &artifacts.Stager{Engine: bin, ModelsDir: models, HFHome: cfg.HFHome, HFEndpoint: cfg.HFEndpoint, Logger: log}
	var offline atomic.Bool
	changes := make(chan struct{}, 1)
	sup, err := engine.New(engine.Config{Engine: bin, Root: replicas, Stager: stager, Offline: offline.Load, StartupTimeout: cfg.StartupTimeout,
		DrainTimeout: cfg.DrainTimeout, Env: cfg.Env, Logger: log, OnChange: func() { poke(changes) }})
	if err != nil {
		return err
	}
	verifier := permit.NewVerifier(keys)
	srv, ln, err := listen(cfg.Listen, creds, pool, id, &proxy.Server{NodeID: id.NodeID, Verifier: verifier, Replay: permit.NewReplay(),
		Backend: proxy.Supervised{S: sup}, Transport: proxy.NewTransport(), Logger: log})
	if err != nil {
		return err
	}
	client, err := enroll.ControlClient(dir, id, creds, 90*time.Second)
	if err != nil {
		return err
	}
	addr := cfg.Advertise
	if addr == "" {
		addr = advertise(ln.Addr())
	}
	runCtx, stop := context.WithCancel(ctx)
	defer stop()
	if cfg.PreventSleep {
		go service.KeepAwake(runCtx, log)
	}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.ServeTLS(ln, "", "") }()
	rep := &reporter.Reporter{Client: client, Logger: log}
	go rep.Run(runCtx)
	go sup.RunMetrics(runCtx, cfg.StatsInterval, rep.Offer)
	collector := &facts.Collector{}
	s := &syncer.Syncer{Client: client, Sup: sup, Verifier: verifier, Dir: dir, Identity: id, Creds: creds, Stager: stager,
		Facts: func(c context.Context) wire.Facts {
			var f wire.Facts
			if cfg.Facts != nil {
				f = cfg.Facts(c)
			} else {
				f = collector.Collect(c, bin, dir.Root)
			}
			if cfg.RankHost != "" {
				f.RankHost = cfg.RankHost
			}
			return f
		}, Address: addr, Offline: &offline,
		Changes: changes, Logger: log}
	log.Info("starlight node running", "node", id.NodeID, "name", id.Name, "listen", ln.Addr().String(), "advertise", addr, "server", id.ServerURL)
	syncDone := make(chan struct{})
	go func() {
		_ = s.Run(runCtx)
		close(syncDone)
	}()
	select {
	case <-ctx.Done():
	case err := <-serveErr:
		stop()
		<-syncDone
		sup.Shutdown(context.Background())
		return fmt.Errorf("inference listener: %w", err)
	}
	stop()
	<-syncDone
	log.Info("draining replicas before exit")
	drain := cfg.DrainTimeout
	if drain <= 0 {
		drain = 2 * time.Minute
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), drain+time.Minute)
	defer cancel()
	sup.Shutdown(shutdownCtx)
	_ = srv.Shutdown(shutdownCtx)
	return nil
}

func poke(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}

// listen opens the gateway-facing mTLS listener; certificate renewals take effect without a restart.
func listen(addr string, creds *state.Creds, pool *x509.CertPool, id state.Identity, h http.Handler) (*http.Server, net.Listener, error) {
	cur, err := creds.Get()
	if err != nil {
		return nil, nil, err
	}
	cfg := pki.ServerConfig(*cur, pool, id.TrustDomain, pki.KindGateway, false)
	cfg.Certificates = nil
	cfg.GetCertificate = func(*tls.ClientHelloInfo) (*tls.Certificate, error) { return creds.Get() }
	if addr == "" {
		addr = ":7443"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, nil, err
	}
	srv := &http.Server{Handler: h, TLSConfig: cfg, ReadHeaderTimeout: 15 * time.Second, IdleTimeout: 2 * time.Minute, MaxHeaderBytes: 64 << 10}
	return srv, ln, nil
}

func resolveEngine(engine string) (string, error) {
	if engine == "" {
		for _, name := range []string{"tensorfold", "tensorfold-native"} {
			if p, err := exec.LookPath(name); err == nil {
				engine = p
				break
			}
		}
		if engine == "" {
			return "", errors.New("no engine found on PATH; pass --engine /path/to/tensorfold")
		}
	}
	p, err := exec.LookPath(engine)
	if err != nil {
		return "", fmt.Errorf("engine %s: %w", engine, err)
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(abs); err != nil {
		return "", err
	}
	return abs, nil
}

// advertise derives host:port from the listener, substituting the hostname for wildcard binds.
func advertise(a net.Addr) string {
	host, port, err := net.SplitHostPort(a.String())
	if err != nil {
		return a.String()
	}
	if ip := net.ParseIP(host); ip == nil || ip.IsUnspecified() {
		if h, err := os.Hostname(); err == nil && h != "" {
			host = h
		}
	}
	return net.JoinHostPort(host, port)
}
