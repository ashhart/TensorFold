// Package core loads Connect's settings, credential store and Starlight clients for the CLI and the tray.
package core

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/appkeys"
	"github.com/ashhart/TensorFold/starlight/internal/connect/bridge"
	"github.com/ashhart/TensorFold/starlight/internal/connect/login"
	"github.com/ashhart/TensorFold/starlight/internal/connect/managed"
	"github.com/ashhart/TensorFold/starlight/internal/connect/profile"
	"github.com/ashhart/TensorFold/starlight/internal/connect/setup"
	"github.com/ashhart/TensorFold/starlight/internal/connect/starlightapi"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
	"github.com/ashhart/TensorFold/starlight/internal/connect/tokens"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

// ErrNoServer means no Starlight address is configured yet.
var ErrNoServer = errors.New("no Starlight server configured; run: tensorfold-connect login --server https://starlight.example")

// Core is Connect's loaded state on this machine.
type Core struct {
	Dir      string
	Profile  profile.Profile
	Settings profile.Settings
	Store    store.Store
	Apps     *appkeys.Registry
	api      *http.Client
	stream   *http.Client
	mu       sync.Mutex
	doc      *discovery.Document
	tokens   *tokens.Manager
}

// Load reads the user's profile and the managed policy from their standard locations.
func Load() (*Core, error) {
	dir, err := profile.Dir()
	if err != nil {
		return nil, err
	}
	p, err := profile.Load(dir)
	if err != nil {
		return nil, err
	}
	pol, err := managed.Load()
	if err != nil {
		return nil, err
	}
	return New(dir, p, pol)
}

// New builds a Core from explicit inputs.
func New(dir string, p profile.Profile, pol managed.Policy) (*Core, error) {
	s, err := profile.Resolve(p, pol)
	if err != nil {
		return nil, err
	}
	tr, err := transport(s.CAPEM)
	if err != nil {
		return nil, err
	}
	apps, err := appkeys.Open(filepath.Join(dir, "apps.json"))
	if err != nil {
		return nil, err
	}
	return &Core{Dir: dir, Profile: p, Settings: s, Store: store.Open(dir), Apps: apps,
		api: &http.Client{Transport: tr, Timeout: 30 * time.Second}, stream: &http.Client{Transport: tr}}, nil
}

func transport(caPEM []byte) (*http.Transport, error) {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	if len(caPEM) > 0 {
		pool, err := x509.SystemCertPool()
		if err != nil || pool == nil {
			pool = x509.NewCertPool()
		}
		if !pool.AppendCertsFromPEM(caPEM) {
			return nil, errors.New("the configured CA certificate could not be loaded")
		}
		t.TLSClientConfig.RootCAs = pool
	}
	return t, nil
}

// Server returns the effective Starlight address.
func (c *Core) Server() (string, error) {
	if c.Settings.Server == "" {
		return "", ErrNoServer
	}
	return c.Settings.Server, nil
}

// UseServer records a server the user chose, unless managed policy fixes it.
func (c *Core) UseServer(raw string) error {
	norm, err := profile.NormalizeServer(raw)
	if err != nil {
		return err
	}
	if norm == c.Settings.Server {
		return nil
	}
	if c.Settings.ServerLocked {
		return fmt.Errorf("your organisation manages the Starlight address (%s); it cannot be changed here", c.Settings.Server)
	}
	c.Profile.Server = norm
	if err := profile.Save(c.Dir, c.Profile); err != nil {
		return err
	}
	c.mu.Lock()
	c.Settings.Server, c.doc, c.tokens = norm, nil, nil
	c.mu.Unlock()
	return nil
}

// UseProject records the project requests act in by default.
func (c *Core) UseProject(slug string) error {
	c.Profile.Project = slug
	c.Settings.Project = slug
	return profile.Save(c.Dir, c.Profile)
}

// Discovery fetches and caches the server's discovery document.
func (c *Core) Discovery(ctx context.Context) (discovery.Document, error) {
	server, err := c.Server()
	if err != nil {
		return discovery.Document{}, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.doc != nil {
		return *c.doc, nil
	}
	doc, err := discovery.Fetch(ctx, c.api, server)
	if err != nil {
		return discovery.Document{}, fmt.Errorf("could not reach Starlight at %s: %w", server, err)
	}
	c.doc = &doc
	return doc, nil
}

// Tokens returns the token manager for the configured server.
func (c *Core) Tokens(ctx context.Context) (*tokens.Manager, error) {
	doc, err := c.Discovery(ctx)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tokens == nil {
		c.tokens = tokens.New(c.Store, filepath.Join(c.Dir, "refresh.lock"), c.api, doc)
	}
	return c.tokens, nil
}

// API returns an authenticated Starlight client.
func (c *Core) API(ctx context.Context) (*starlightapi.Client, error) {
	tm, err := c.Tokens(ctx)
	if err != nil {
		return nil, err
	}
	doc, _ := c.Discovery(ctx)
	return &starlightapi.Client{HTTP: c.api, Doc: doc, Tokens: tm}, nil
}

// LoginOptions select the sign-in flow.
type LoginOptions struct {
	Device bool
	Open   func(string) error
	Out    io.Writer
}

// Login signs in, stores the tokens and returns the signed-in user.
func (c *Core) Login(ctx context.Context, o LoginOptions) (discovery.Me, error) {
	doc, err := c.Discovery(ctx)
	if err != nil {
		return discovery.Me{}, err
	}
	lo := login.Options{Client: c.api, Open: o.Open, Out: o.Out, DeviceName: DeviceName()}
	var tr discovery.TokenResponse
	if o.Device {
		tr, err = login.Device(ctx, doc, lo)
	} else {
		tr, err = login.Browser(ctx, doc, lo)
	}
	if err != nil {
		return discovery.Me{}, err
	}
	tm, err := c.Tokens(ctx)
	if err != nil {
		return discovery.Me{}, err
	}
	if err := tm.Save(tr); err != nil {
		return discovery.Me{}, fmt.Errorf("store sign-in in %s: %w", c.Store.Backend(), err)
	}
	api, err := c.API(ctx)
	if err != nil {
		return discovery.Me{}, err
	}
	return api.Me(ctx)
}

// Logout revokes and forgets the sign-in.
func (c *Core) Logout(ctx context.Context) error {
	if c.Settings.Server == "" {
		return c.Store.Delete(tokens.Key)
	}
	tm, err := c.Tokens(ctx)
	if err != nil {
		return c.Store.Delete(tokens.Key)
	}
	return tm.SignOut(ctx)
}

// DeviceName labels this machine in Starlight's list of signed-in devices.
func DeviceName() string {
	h, err := os.Hostname()
	if err != nil || h == "" {
		h = "unknown"
	}
	h, _, _ = strings.Cut(h, ".")
	if len(h) > 48 {
		h = h[:48]
	}
	return "TensorFold Connect on " + h
}

// BridgeURL is the local endpoint's base address for port, or for the configured port when port is 0.
func (c *Core) BridgeURL(port int) string {
	if port <= 0 {
		port = c.Settings.BridgePort
	}
	return fmt.Sprintf("http://127.0.0.1:%d", port)
}

// Bridge prepares the local endpoint; port 0 takes the configured port.
func (c *Core) Bridge(ctx context.Context, port int) (*bridge.Server, error) {
	if c.Settings.BridgeDisabled {
		return nil, errors.New("your organisation has turned off the local endpoint")
	}
	if port <= 0 {
		port = c.Settings.BridgePort
	}
	doc, err := c.Discovery(ctx)
	if err != nil {
		return nil, err
	}
	tm, err := c.Tokens(ctx)
	if err != nil {
		return nil, err
	}
	return bridge.New(bridge.Config{Port: port, Upstream: doc.Issuer, HTTP: c.stream, Tokens: tm, Apps: c.Apps,
		AllowedOrigins: c.Settings.AllowedOrigins, Project: c.Settings.Project})
}

// BridgeRunning reports whether a local endpoint answers on the configured port.
func (c *Core) BridgeRunning(ctx context.Context, port int) bool {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BridgeURL(port)+"/healthz", nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Setup issues a fresh key for the tool's app and renders its configuration; replaced reports a rotated key.
func (c *Core) Setup(tool, app, model, shell string, port int) (text string, replaced bool, err error) {
	if app == "" {
		app = tool
	}
	check := setup.Options{Tool: tool, BaseURL: c.BridgeURL(port), Key: "slc_check", Model: model, Shell: shell}
	if _, err := setup.Render(check); err != nil {
		return "", false, err
	}
	apps, err := c.Apps.List()
	if err != nil {
		return "", false, err
	}
	for _, a := range apps {
		replaced = replaced || a.Name == app
	}
	key, err := c.Apps.Replace(app)
	if err != nil {
		return "", false, err
	}
	text, err = setup.Render(setup.Options{Tool: tool, BaseURL: c.BridgeURL(port), Key: key, Model: model, Shell: shell})
	return text, replaced, err
}
