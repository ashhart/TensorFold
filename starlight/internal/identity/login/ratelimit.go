package login

import (
	"net/http"

	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ratelimit"
)

// limiter slows floods per client network, counted across every Starlight instance.
type limiter struct {
	shared  *ratelimit.Limiter
	proxies *httpx.Proxies
}

func newLimiter(cfg Config, name string, perMinute int) *limiter {
	return &limiter{shared: ratelimit.Shared(cfg.Pool, cfg.RateKey, name, float64(perMinute)/60, perMinute), proxies: cfg.Proxies}
}

// allow spends one attempt for the request's client network and reports whether one was available.
func (l *limiter) allow(r *http.Request) bool {
	return l.shared.Allow(ratelimit.Client(l.proxies.ClientIP(r)))
}
