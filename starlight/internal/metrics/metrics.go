// Package metrics defines Starlight's Prometheus series; labels are bounded by the catalogue, never by users.
package metrics

import (
	"net/http"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
)

// Set is every series Starlight exports.
type Set struct {
	Registry    *prometheus.Registry
	Requests    *prometheus.CounterVec
	Duration    *prometheus.HistogramVec
	FirstByte   *prometheus.HistogramVec
	QueueWait   *prometheus.HistogramVec
	Tokens      *prometheus.CounterVec
	Rejections  *prometheus.CounterVec
	InFlight    *prometheus.GaugeVec
	QueueDepth  *prometheus.GaugeVec
	Replicas    *prometheus.GaugeVec
	Nodes       *prometheus.GaugeVec
	LeaderEpoch prometheus.Gauge
	CertExpiry  *prometheus.GaugeVec
}

var latency = []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300, 600}

// New registers the series with Go and process collectors.
func New() *Set {
	r := prometheus.NewRegistry()
	r.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	s := &Set{
		Registry: r,
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{Name: "starlight_requests_total", Help: "Inference requests by outcome."},
			[]string{"model", "protocol", "route", "status", "outcome"}),
		Duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "starlight_request_duration_seconds", Help: "Admission to last byte.", Buckets: latency},
			[]string{"model", "route"}),
		FirstByte: prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "starlight_first_byte_seconds", Help: "Arrival to first byte sent.", Buckets: latency},
			[]string{"model", "route"}),
		QueueWait: prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "starlight_queue_wait_seconds", Help: "Time spent waiting for capacity.", Buckets: latency},
			[]string{"model", "priority"}),
		Tokens: prometheus.NewCounterVec(prometheus.CounterOpts{Name: "starlight_tokens_total", Help: "Tokens by direction."},
			[]string{"model", "route", "direction"}),
		Rejections: prometheus.NewCounterVec(prometheus.CounterOpts{Name: "starlight_admission_rejections_total", Help: "Requests refused before dispatch."},
			[]string{"reason"}),
		InFlight: prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "starlight_inflight_requests", Help: "Requests being served by this instance."},
			[]string{"route"}),
		QueueDepth: prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "starlight_queue_depth", Help: "Requests waiting for capacity, all instances."},
			[]string{"model"}),
		Replicas: prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "starlight_replicas", Help: "Replicas by deployment and state."},
			[]string{"deployment", "state"}),
		Nodes:       prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "starlight_nodes", Help: "Nodes by status."}, []string{"status"}),
		LeaderEpoch: prometheus.NewGauge(prometheus.GaugeOpts{Name: "starlight_controller_epoch", Help: "Fencing epoch held by this instance, 0 if not leader."}),
		CertExpiry: prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "starlight_certificate_expiry_timestamp_seconds",
			Help: "When the public certificate and the workload CA expire, as Unix time."}, []string{"certificate"}),
	}
	r.MustRegister(s.Requests, s.Duration, s.FirstByte, s.QueueWait, s.Tokens, s.Rejections, s.InFlight, s.QueueDepth,
		s.Replicas, s.Nodes, s.LeaderEpoch, s.CertExpiry)
	return s
}

// Observe records one finished request.
func (s *Set) Observe(model, protocol, route string, status int, outcome string, total, first time.Duration, prompt, completion, cached int64) {
	s.Requests.WithLabelValues(model, protocol, route, statusClass(status), outcome).Inc()
	s.Duration.WithLabelValues(model, route).Observe(total.Seconds())
	if first > 0 {
		s.FirstByte.WithLabelValues(model, route).Observe(first.Seconds())
	}
	s.Tokens.WithLabelValues(model, route, "prompt").Add(float64(prompt))
	s.Tokens.WithLabelValues(model, route, "completion").Add(float64(completion))
	s.Tokens.WithLabelValues(model, route, "cached").Add(float64(cached))
}

func statusClass(code int) string {
	switch {
	case code >= 500:
		return "5xx"
	case code >= 400:
		return "4xx"
	case code >= 200:
		return "2xx"
	}
	return "other"
}

// Handler serves the registry, requiring a bearer token when one is configured.
func (s *Set) Handler(token string) http.Handler {
	h := promhttp.HandlerFor(s.Registry, promhttp.HandlerOpts{EnableOpenMetrics: true})
	if token == "" {
		return h
	}
	want := []byte("Bearer " + strings.TrimSpace(token))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !secret.Equal([]byte(r.Header.Get("Authorization")), want) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="metrics"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		h.ServeHTTP(w, r)
	})
}
