package httpx

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"net"
	"net/http"
	"net/netip"
	"strings"
)

type ctxKey int

const requestIDKey ctxKey = 1

// RequestID returns the id assigned by WithRequestID.
func RequestID(ctx context.Context) string {
	s, _ := ctx.Value(requestIDKey).(string)
	return s
}

// NewRequestID returns a random 128-bit hex id.
func NewRequestID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return "req_" + hex.EncodeToString(b)
}

// WithRequestID assigns every request a fresh id and echoes it; client-sent ids are never trusted.
func WithRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := NewRequestID()
		w.Header().Set("X-Request-Id", id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), requestIDKey, id)))
	})
}

// Proxies resolves the client address, honouring X-Forwarded-For only from trusted hops.
type Proxies struct {
	trusted []netip.Prefix
}

// NewProxies parses CIDRs of reverse proxies whose forwarding headers are believed.
func NewProxies(cidrs []string) (*Proxies, error) {
	p := &Proxies{}
	for _, c := range cidrs {
		pfx, err := netip.ParsePrefix(strings.TrimSpace(c))
		if err != nil {
			return nil, err
		}
		p.trusted = append(p.trusted, pfx.Masked())
	}
	return p, nil
}

func (p *Proxies) isTrusted(a netip.Addr) bool {
	for _, t := range p.trusted {
		if t.Contains(a) {
			return true
		}
	}
	return false
}

// ClientIP walks X-Forwarded-For from the right, skipping trusted proxies; an unreadable hop yields the zero Addr.
func (p *Proxies) ClientIP(r *http.Request) netip.Addr {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	addr, err := netip.ParseAddr(host)
	if err != nil {
		return netip.Addr{}
	}
	addr = addr.Unmap()
	if p == nil || !p.isTrusted(addr) {
		return addr
	}
	hops := r.Header.Values("X-Forwarded-For")
	var all []string
	for _, h := range hops {
		all = append(all, strings.Split(h, ",")...)
	}
	for i := len(all) - 1; i >= 0; i-- {
		a, ok := hopAddr(strings.TrimSpace(all[i]))
		if !ok {
			return netip.Addr{}
		}
		if !p.isTrusted(a) {
			return a
		}
		addr = a
	}
	return addr
}

// hopAddr reads one X-Forwarded-For entry: a bare address, or ip:port and [v6]:port as Azure Application Gateway sends.
func hopAddr(s string) (netip.Addr, bool) {
	if a, err := netip.ParseAddr(s); err == nil {
		return a.Unmap(), true
	}
	if ap, err := netip.ParseAddrPort(s); err == nil {
		return ap.Addr().Unmap(), true
	}
	return netip.Addr{}, false
}

// ReadBody reads at most limit bytes and reports ErrTooLarge beyond that.
func ReadBody(r *http.Request, limit int64) ([]byte, error) {
	body, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return nil, ErrTooLarge
		}
		return nil, Errorf(http.StatusBadRequest, "invalid_body", "could not read request body")
	}
	if int64(len(body)) > limit {
		return nil, ErrTooLarge
	}
	return body, nil
}

// BearerToken extracts a bearer or x-api-key credential.
func BearerToken(r *http.Request) string {
	if h := r.Header.Get("Authorization"); h != "" {
		scheme, tok, ok := strings.Cut(h, " ")
		if ok && strings.EqualFold(scheme, "Bearer") {
			return strings.TrimSpace(tok)
		}
		return ""
	}
	return strings.TrimSpace(r.Header.Get("X-Api-Key"))
}

// SecureHeaders sets defensive headers for browser-facing responses.
func SecureHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "same-origin")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Cross-Origin-Opener-Policy", "same-origin")
		h.Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
		if r.TLS != nil {
			h.Set("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
		}
		next.ServeHTTP(w, r)
	})
}
