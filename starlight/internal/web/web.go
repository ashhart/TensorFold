// Package web renders Starlight's server-side pages with a shared layout, brand stylesheet and logo.
package web

import (
	"bytes"
	"context"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"html/template"
	"io/fs"
	"log/slog"
	"mime"
	"net/http"
	"path"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

//go:embed templates/*.html
var templateFS embed.FS

//go:embed static/*
var staticFS embed.FS

// CSP allows only same-origin assets; it omits form-action so sign-in forms may redirect to identity providers.
const CSP = "default-src 'none'; img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'"

// ConsoleCSP adds same-origin fetches, which the live dashboard uses to poll its own JSON.
const ConsoleCSP = CSP + "; connect-src 'self'"

// Pages the renderer knows.
const (
	PageLogin   = "login"
	PageMessage = "message"
	PageApprove = "approve"
	PageDevice  = "device"
	PageSetup   = "setup"
	PageRecover = "recover"
)

var assetVersions = map[string]string{}

func init() {
	entries, err := fs.ReadDir(staticFS, "static")
	if err != nil {
		panic(err)
	}
	for _, e := range entries {
		b, err := staticFS.ReadFile("static/" + e.Name())
		if err != nil {
			panic(err)
		}
		sum := sha256.Sum256(b)
		assetVersions[e.Name()] = hex.EncodeToString(sum[:4])
	}
}

// Asset returns the cache-busting URL of a static file.
func Asset(name string) string {
	return "/static/" + name + "?v=" + assetVersions[name]
}

// Renderer executes page templates inside the layout.
type Renderer struct {
	pages map[string]*template.Template
	org   func(context.Context) string
}

// New parses every page; org supplies the organisation name shown in the header.
func New(org func(context.Context) string) (*Renderer, error) {
	if org == nil {
		org = func(context.Context) string { return "TensorFold Starlight" }
	}
	r := &Renderer{pages: map[string]*template.Template{}, org: org}
	funcs := template.FuncMap{"asset": Asset}
	for _, p := range []string{PageLogin, PageMessage, PageApprove, PageDevice, PageSetup, PageRecover} {
		t, err := template.New("layout").Funcs(funcs).ParseFS(templateFS, "templates/layout.html", "templates/"+p+".html")
		if err != nil {
			return nil, fmt.Errorf("web: %s: %w", p, err)
		}
		r.pages[p] = t
	}
	return r, nil
}

type view struct {
	Title     string
	Org       string
	RequestID string
	Page      any
}

// Render writes page with data and status, or a plain 500 when the page itself fails.
func (r *Renderer) Render(w http.ResponseWriter, req *http.Request, status int, page, title string, data any) {
	t, ok := r.pages[page]
	if !ok {
		http.Error(w, "unknown page", http.StatusInternalServerError)
		return
	}
	var buf bytes.Buffer
	v := view{Title: title, Org: r.org(req.Context()), RequestID: httpx.RequestID(req.Context()), Page: data}
	if err := t.ExecuteTemplate(&buf, "layout", v); err != nil {
		slog.Error("render page", "page", page, "err", err)
		http.Error(w, "page failed to render", http.StatusInternalServerError)
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/html; charset=utf-8")
	h.Set("Cache-Control", "no-store")
	h.Set("Content-Security-Policy", CSP)
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("X-Frame-Options", "DENY")
	h.Set("Referrer-Policy", "same-origin")
	w.WriteHeader(status)
	_, _ = w.Write(buf.Bytes())
}

// Message renders the generic message page.
func (r *Renderer) Message(w http.ResponseWriter, req *http.Request, status int, kind, heading, body string, links ...Link) {
	r.Render(w, req, status, PageMessage, heading, MessagePage{Kind: kind, Heading: heading, Body: body, Links: links})
}

// StaticHandler serves the embedded stylesheet and images under /static/.
func StaticHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(path.Clean(r.URL.Path), "/static/")
		if _, ok := assetVersions[name]; !ok || strings.Contains(name, "/") {
			http.NotFound(w, r)
			return
		}
		b, err := staticFS.ReadFile("static/" + name)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		ct := mime.TypeByExtension(path.Ext(name))
		if ct == "" {
			ct = "application/octet-stream"
		}
		w.Header().Set("Content-Type", ct)
		w.Header().Set("X-Content-Type-Options", "nosniff")
		if r.URL.Query().Get("v") == assetVersions[name] {
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		} else {
			w.Header().Set("Cache-Control", "public, max-age=300")
		}
		_, _ = w.Write(b)
	})
}
