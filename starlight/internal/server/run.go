package server

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/login"
	"github.com/ashhart/TensorFold/starlight/internal/mdns"
	"github.com/ashhart/TensorFold/starlight/internal/scheduler"
)

const discoveryPath = discovery.Path

var netipZero = netip.Addr{}

func hashToken(t string) []byte { return secret.Hash(t) }

func (t *Starlight) discovery(w http.ResponseWriter, r *http.Request) {
	httpx.JSON(w, http.StatusOK, discovery.Build(t.Config.PublicURL, t.Inst.OrgName, version(), t.Inst.ID, t.Policies.Offline(r.Context())))
}

// Run serves until ctx ends, then drains listeners for up to 30 seconds.
func (t *Starlight) Run(ctx context.Context) error {
	var wg sync.WaitGroup
	bg := func(f func(context.Context)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f(ctx)
		}()
	}
	bg(t.Trust.Run)
	bg(func(ctx context.Context) { refreshEgress(ctx, t.Pool, t.Egress) })
	if t.Config.HasRole("gateway") {
		bg(func(ctx context.Context) { admission.Heartbeat(ctx, t.Pool, t.Instance, hostname(), version()) })
		bg(func(ctx context.Context) { t.Admitter.Waker.Run(ctx, t.Pool) })
	}
	if t.Config.HasRole("control") {
		bg(t.Control.Listen)
		ctrl := &scheduler.Controller{Pool: t.Pool, Instance: t.Instance, Log: t.Log,
			Hourly: []func(context.Context) error{func(ctx context.Context) error { return login.Cleanup(ctx, t.Pool) }}}
		elector := &scheduler.Elector{Pool: t.Pool, Instance: t.Instance}
		bg(func(ctx context.Context) {
			elector.Run(ctx, func(ctx context.Context, epoch int64) {
				t.Metrics.LeaderEpoch.Set(float64(epoch))
				defer t.Metrics.LeaderEpoch.Set(0)
				ctrl.Lead(ctx, epoch)
			})
		})
	}
	if t.Config.MDNS.Advertise {
		bg(t.advertise)
	}
	bg(t.gauges)
	servers, err := t.listen()
	if err != nil {
		return err
	}
	errc := make(chan error, len(servers))
	for _, s := range servers {
		go func(s *served) { errc <- s.serve() }(s)
	}
	t.Log.Info("starlight started", "version", version(), "instance", t.Instance, "roles", strings.Join(t.Config.Roles, ","), "public", t.Config.PublicURL)
	var runErr error
	select {
	case <-ctx.Done():
	case runErr = <-errc:
	}
	shut, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, s := range servers {
		_ = s.srv.Shutdown(shut)
	}
	wg.Wait()
	if errors.Is(runErr, http.ErrServerClosed) {
		runErr = nil
	}
	return runErr
}

type served struct {
	srv *http.Server
	ln  net.Listener
	tls bool
}

func (s *served) serve() error {
	if s.tls {
		return s.srv.ServeTLS(s.ln, "", "")
	}
	return s.srv.Serve(s.ln)
}

func (t *Starlight) listen() ([]*served, error) {
	var out []*served
	public := &http.Server{Handler: t.Handler(), ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 2 * time.Minute, MaxHeaderBytes: 64 << 10}
	useTLS := t.Config.TLS.CertFile != ""
	if useTLS {
		cert, err := loadCertFile(t.Config.TLS.CertFile, t.Config.TLS.KeyFile)
		if err != nil {
			return nil, err
		}
		t.publicCert = cert
		public.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12, GetCertificate: cert.get}
	}
	ln, err := net.Listen("tcp", t.Config.Listen.Public)
	if err != nil {
		return nil, err
	}
	out = append(out, &served{srv: public, ln: ln, tls: useTLS})
	if t.Config.HasRole("control") {
		nodes := &http.Server{Handler: t.Control.Handler(), TLSConfig: t.Control.TLSConfig(), ReadHeaderTimeout: 10 * time.Second,
			WriteTimeout: 2 * time.Minute, IdleTimeout: 2 * time.Minute, MaxHeaderBytes: 32 << 10}
		nl, err := net.Listen("tcp", t.Config.Listen.Nodes)
		if err != nil {
			return nil, err
		}
		out = append(out, &served{srv: nodes, ln: nl, tls: true})
	}
	token := ""
	if f := t.Config.Metrics.BearerTokenFile; f != "" {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		token = strings.TrimSpace(string(b))
	}
	mux := http.NewServeMux()
	mux.Handle("GET /metrics", t.Metrics.Handler(token))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /readyz", t.ready)
	ml, err := net.Listen("tcp", t.Config.Listen.Metrics)
	if err != nil {
		return nil, err
	}
	out = append(out, &served{srv: &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}, ln: ml})
	return out, nil
}

// gauges publishes fleet-wide state every 15 seconds; every instance reports the same values from Postgres.
func (t *Starlight) gauges(ctx context.Context) {
	tick := time.NewTicker(15 * time.Second)
	defer tick.Stop()
	for {
		t.sample(ctx)
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}

func (t *Starlight) sample(ctx context.Context) {
	set := func(g interface{ Reset() }, sql string, apply func(labels []string, v float64)) {
		rows, err := t.Pool.Query(ctx, sql)
		if err != nil {
			return
		}
		defer rows.Close()
		g.Reset()
		for rows.Next() {
			var a, b string
			var v float64
			if rows.Scan(&a, &b, &v) == nil {
				apply([]string{a, b}, v)
			}
		}
	}
	m := t.Metrics
	set(m.Nodes, `SELECT CASE WHEN status = 'active' AND (last_seen_at IS NULL OR last_seen_at < now() - interval '90 seconds') THEN 'stale'
		ELSE status END, '', count(*)::float8 FROM nodes GROUP BY 1`, func(l []string, v float64) { m.Nodes.WithLabelValues(l[0]).Set(v) })
	set(m.Replicas, `SELECT d.name, r.state, count(*)::float8 FROM replicas r JOIN deployments d ON d.id = r.deployment_id GROUP BY 1, 2`,
		func(l []string, v float64) { m.Replicas.WithLabelValues(l[0], l[1]).Set(v) })
	set(m.QueueDepth, `SELECT m.name, '', count(*)::float8 FROM admission_queue q JOIN models m ON m.id = q.model_id
		WHERE q.heartbeat_at > now() - interval '30 seconds' AND q.deadline > now() GROUP BY 1`,
		func(l []string, v float64) { m.QueueDepth.WithLabelValues(l[0]).Set(v) })
	if ca := t.Trust.CA(); ca != nil {
		m.CertExpiry.WithLabelValues("workload_ca").Set(float64(ca.Cert.NotAfter.Unix()))
	}
	if t.publicCert != nil {
		m.CertExpiry.WithLabelValues("public").Set(float64(t.publicCert.notAfter().Unix()))
	}
}

func (t *Starlight) advertise(ctx context.Context) {
	u, err := url.Parse(t.Config.PublicURL)
	if err != nil {
		return
	}
	port := 443
	if p := u.Port(); p != "" {
		port, _ = strconv.Atoi(p)
	} else if u.Scheme == "http" {
		port = 80
	}
	instance := t.Config.MDNS.Instance
	if instance == "" {
		instance = t.Inst.OrgName
	}
	stop, err := mdns.Advertise(ctx, mdns.Config{Instance: instance, Port: port, URL: t.Config.PublicURL, InstallationID: t.Inst.Short()})
	if err != nil {
		t.Log.Warn("mdns advertisement failed", "error", err)
		return
	}
	<-ctx.Done()
	stop()
}
