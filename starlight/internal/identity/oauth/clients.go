// Package oauth is Starlight's authorisation server for its own public clients: Connect and CLIs.
package oauth

import (
	"net"
	"net/url"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

// Client is a built-in public client; none has a secret, so PKCE or the device grant is mandatory.
type Client struct {
	ID          string
	Name        string
	Loopback    bool
	DeviceGrant bool
	Scopes      []string
}

var clients = map[string]Client{
	discovery.ClientConnect: {ID: discovery.ClientConnect, Name: "TensorFold Connect", Loopback: true, DeviceGrant: true,
		Scopes: []string{discovery.ScopeInference, discovery.ScopeOffline, discovery.ScopeProfile}},
	discovery.ClientCLI: {ID: discovery.ClientCLI, Name: "TensorFold CLI", DeviceGrant: true,
		Scopes: []string{discovery.ScopeInference, discovery.ScopeOffline, discovery.ScopeProfile}},
}

// LookupClient returns a built-in client.
func LookupClient(id string) (Client, bool) {
	c, ok := clients[id]
	return c, ok
}

// ValidRedirect accepts only RFC 8252 loopback IP redirects on any port, never "localhost" or other hosts.
func (c Client) ValidRedirect(raw string) bool {
	if !c.Loopback {
		return false
	}
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "http" || u.User != nil || u.Fragment != "" || u.RawQuery != "" {
		return false
	}
	ip := net.ParseIP(u.Hostname())
	if ip == nil || !ip.IsLoopback() || u.Port() == "" {
		return false
	}
	return u.Path == "" || strings.HasPrefix(u.Path, "/")
}

// NarrowScope intersects a requested space-separated scope with what the client may hold.
func (c Client) NarrowScope(requested string) (string, bool) {
	if strings.TrimSpace(requested) == "" {
		return strings.Join([]string{discovery.ScopeInference, discovery.ScopeProfile}, " "), true
	}
	var out []string
	seen := map[string]bool{}
	for _, s := range strings.Fields(requested) {
		allowed := false
		for _, a := range c.Scopes {
			allowed = allowed || a == s
		}
		if !allowed {
			return "", false
		}
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return strings.Join(out, " "), true
}

func hasScope(scope, s string) bool {
	for _, x := range strings.Fields(scope) {
		if x == s {
			return true
		}
	}
	return false
}

// DeviceName keeps the client-supplied label short and printable for consent screens and audit.
func DeviceName(s string) string {
	s = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, strings.TrimSpace(s))
	if len(s) > 64 {
		s = s[:64]
	}
	return s
}
