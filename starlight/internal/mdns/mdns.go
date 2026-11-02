// Package mdns advertises Starlight on the local link and finds it; results are hints for the user, never trust.
package mdns

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	hmdns "github.com/hashicorp/mdns"
)

// Service is the DNS-SD service type Starlight advertises.
const Service = "_tf-starlight._tcp"

// Config is what a Starlight instance advertises; it carries no secrets.
type Config struct {
	Instance       string
	Port           int
	URL            string
	InstallationID string
	IPs            []net.IP
}

// Found is one advertised Starlight.
type Found struct {
	Instance       string
	Host           string
	Addrs          []net.IP
	Port           int
	URL            string
	InstallationID string
}

var (
	quiet   = log.New(io.Discard, "", 0)
	idChars = regexp.MustCompile(`^[0-9a-z]{1,8}$`)
	label   = regexp.MustCompile(`[^a-z0-9-]+`)
)

// TXT builds the advertised records: a version, the canonical URL and a short installation id.
func TXT(cfg Config) ([]string, error) {
	u, err := url.Parse(cfg.URL)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" || u.User != nil {
		return nil, fmt.Errorf("advertised URL %q must be an absolute http or https URL", cfg.URL)
	}
	if len(cfg.URL) > 240 {
		return nil, errors.New("advertised URL is longer than a TXT string allows")
	}
	id := strings.ToLower(strings.ReplaceAll(cfg.InstallationID, "-", ""))
	if len(id) > 8 {
		id = id[:8]
	}
	if !idChars.MatchString(id) {
		return nil, fmt.Errorf("installation id %q has no usable prefix", cfg.InstallationID)
	}
	return []string{"v=1", "url=" + cfg.URL, "id=" + id}, nil
}

// Advertise answers mDNS queries for Starlight until ctx ends or stop is called.
func Advertise(ctx context.Context, cfg Config) (stop func(), err error) {
	if cfg.Port <= 0 || cfg.Port > 65535 {
		return nil, fmt.Errorf("advertised port %d is out of range", cfg.Port)
	}
	txt, err := TXT(cfg)
	if err != nil {
		return nil, err
	}
	ips := cfg.IPs
	if len(ips) == 0 {
		if ips, err = localIPs(); err != nil {
			return nil, err
		}
	}
	svc, err := hmdns.NewMDNSService(instanceName(cfg.Instance), Service, "", hostName(), cfg.Port, ips, txt)
	if err != nil {
		return nil, err
	}
	srv, err := hmdns.NewServer(&hmdns.Config{Zone: svc, Logger: quiet})
	if err != nil {
		return nil, err
	}
	var once sync.Once
	stop = func() { once.Do(func() { _ = srv.Shutdown() }) }
	go func() {
		<-ctx.Done()
		stop()
	}()
	return stop, nil
}

// Browse collects the Starlight servers that answer within timeout.
func Browse(ctx context.Context, timeout time.Duration) ([]Found, error) {
	params := hmdns.DefaultParams(Service)
	params.Timeout, params.Logger = timeout, quiet
	return collect(func(entries chan<- *hmdns.ServiceEntry) error {
		params.Entries = entries
		return hmdns.QueryContext(ctx, params)
	})
}

// collect reads entries after query returns, since the client keeps updating an entry it already sent.
func collect(query func(chan<- *hmdns.ServiceEntry) error) ([]Found, error) {
	entries := make(chan *hmdns.ServiceEntry, 64)
	var got []*hmdns.ServiceEntry
	done := make(chan struct{})
	go func() {
		defer close(done)
		for e := range entries {
			got = append(got, e)
		}
	}()
	err := query(entries)
	close(entries)
	<-done
	var out []Found
	seen := map[string]bool{}
	for _, e := range got {
		f, ok := parse(e)
		if !ok || seen[f.URL+"|"+f.InstallationID] {
			continue
		}
		seen[f.URL+"|"+f.InstallationID] = true
		out = append(out, f)
	}
	return out, err
}

func parse(e *hmdns.ServiceEntry) (Found, bool) {
	f := Found{Host: e.Host, Port: e.Port, Instance: strings.TrimSuffix(strings.TrimSuffix(e.Name, "."), "."+Service+".local")}
	version := ""
	for _, field := range e.InfoFields {
		k, v, _ := strings.Cut(field, "=")
		switch k {
		case "v":
			version = v
		case "url":
			f.URL = v
		case "id":
			f.InstallationID = v
		}
	}
	u, err := url.Parse(f.URL)
	if version != "1" || err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return Found{}, false
	}
	if e.AddrV4 != nil {
		f.Addrs = append(f.Addrs, e.AddrV4)
	}
	if e.AddrV6 != nil {
		f.Addrs = append(f.Addrs, e.AddrV6)
	}
	return f, true
}

func instanceName(s string) string {
	s = strings.Trim(label.ReplaceAllString(strings.ToLower(s), "-"), "-")
	if s == "" {
		s = "tensorfold-starlight"
	}
	if len(s) > 63 {
		s = s[:63]
	}
	return s
}

func hostName() string {
	h, err := os.Hostname()
	if err != nil {
		h = "starlight"
	}
	first, _, _ := strings.Cut(strings.ToLower(h), ".")
	first = strings.Trim(label.ReplaceAllString(first, "-"), "-")
	if first == "" {
		first = "starlight"
	}
	return first + ".local."
}

// localIPs lists unicast addresses on interfaces that are up and multicast-capable.
func localIPs() ([]net.IP, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	var out []net.IP
	for _, ifc := range ifaces {
		if ifc.Flags&net.FlagUp == 0 || ifc.Flags&net.FlagMulticast == 0 || ifc.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := ifc.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			ipn, ok := a.(*net.IPNet)
			if !ok {
				continue
			}
			if ipn.IP.IsGlobalUnicast() || (ipn.IP.IsLinkLocalUnicast() && ipn.IP.To4() == nil) {
				out = append(out, ipn.IP)
			}
		}
	}
	if len(out) == 0 {
		return nil, errors.New("no multicast-capable network interface with an address")
	}
	return out, nil
}
