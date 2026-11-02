// Package providers sends approved requests to frontier APIs through an allowlisted, offline-aware client.
package providers

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ErrOffline means the installation forbids all egress.
var ErrOffline = errors.New("providers: this installation is offline")

// Egress is the only HTTP client allowed to leave the network; it refuses hosts nobody approved.
type Egress struct {
	mu       sync.RWMutex
	hosts    map[string]bool
	private  map[string]bool
	offline  bool
	client   *http.Client
	resolver *net.Resolver
	proxies  map[string]bool
}

// NewEgress builds the client; proxies come from HTTPS_PROXY and NO_PROXY as usual.
func NewEgress() *Egress {
	e := &Egress{hosts: map[string]bool{}, private: map[string]bool{}, resolver: net.DefaultResolver, proxies: proxyHosts()}
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second, ControlContext: e.control}
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          256,
		MaxIdleConnsPerHost:   64,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 10 * time.Minute,
	}
	e.client = &http.Client{Transport: &guarded{e: e, next: transport}, CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	return e
}

// Allow replaces the approved host set; private marks hosts allowed to resolve to internal addresses.
func (e *Egress) Allow(hosts map[string]bool, offline bool) {
	pub, priv := map[string]bool{}, map[string]bool{}
	for h, internal := range hosts {
		pub[strings.ToLower(h)] = true
		if internal {
			priv[strings.ToLower(h)] = true
		}
	}
	e.mu.Lock()
	e.hosts, e.private, e.offline = pub, priv, offline
	e.mu.Unlock()
}

// Client returns the guarded client.
func (e *Egress) Client() *http.Client { return e.client }

type guarded struct {
	e    *Egress
	next http.RoundTripper
}

type hostKey struct{}

func (g *guarded) RoundTrip(r *http.Request) (*http.Response, error) {
	g.e.mu.RLock()
	offline, ok := g.e.offline, g.e.hosts[strings.ToLower(r.URL.Hostname())]
	g.e.mu.RUnlock()
	if offline {
		return nil, ErrOffline
	}
	if !ok {
		return nil, fmt.Errorf("providers: egress to %q is not approved", r.URL.Hostname())
	}
	if r.URL.Scheme != "https" && !g.e.isPrivate(r.URL.Hostname()) {
		return nil, errors.New("providers: cloud egress must use https")
	}
	ctx := context.WithValue(r.Context(), hostKey{}, strings.ToLower(r.URL.Hostname()))
	return g.next.RoundTrip(r.WithContext(ctx))
}

func (e *Egress) isPrivate(host string) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.private[strings.ToLower(host)]
}

// control refuses internal addresses unless this request's host is an internal provider or the dial is to the proxy.
func (e *Egress) control(ctx context.Context, network, address string, _ syscall.RawConn) error {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return err
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return fmt.Errorf("providers: unresolved dial target %q", address)
	}
	if ip.IsUnspecified() || ip.IsMulticast() || ip.Equal(net.ParseIP("169.254.169.254")) {
		return fmt.Errorf("providers: refusing address %s", ip)
	}
	if !internalIP(ip) || e.proxies[address] {
		return nil
	}
	target, _ := ctx.Value(hostKey{}).(string)
	if target != "" && e.isPrivate(target) {
		return nil
	}
	return fmt.Errorf("providers: refusing internal address %s for %q", ip, target)
}

// proxyHosts lists the HTTPS proxy addresses from the environment, which egress may always dial.
func proxyHosts() map[string]bool {
	out := map[string]bool{}
	req := &http.Request{URL: &url.URL{Scheme: "https", Host: "provider.invalid"}}
	if u, err := http.ProxyFromEnvironment(req); err == nil && u != nil {
		port := u.Port()
		if port == "" {
			port = "443"
			if u.Scheme == "http" {
				port = "80"
			}
		}
		if ips, err := net.LookupIP(u.Hostname()); err == nil {
			for _, ip := range ips {
				out[net.JoinHostPort(ip.String(), port)] = true
			}
		}
	}
	return out
}

func internalIP(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsInterfaceLocalMulticast()
}

// HostOf returns the lowercase host of a base URL, for building the allowlist.
func HostOf(base string) (string, error) {
	u, err := url.Parse(base)
	if err != nil || u.Host == "" {
		return "", fmt.Errorf("provider base url %q is not absolute", base)
	}
	return strings.ToLower(u.Hostname()), nil
}
