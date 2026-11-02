// Package console serves the administrator web interface; every page and action checks the signed-in user's grants.
package console

import (
	"context"
	"embed"
	"fmt"
	"html/template"
	"io/fs"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/login"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

//go:embed templates
var templateFS embed.FS

// Console holds what pages need; Flush clears every cache after an edit so it applies at once on this instance.
type Console struct {
	Pool      *pgxpool.Pool
	Keys      *keyring.Keyring
	Sessions  *session.Manager
	Login     *login.Handler
	Catalog   *catalog.Catalog
	Trust     *trust.Trust
	Proxies   *httpx.Proxies
	PublicURL string
	NodeURL   string
	Version   string
	Org       func(context.Context) string
	Flush     func()
	pages     map[string]*template.Template
	mux       *http.ServeMux
}

// Handler is one console action.
type Handler func(q *Request) error

// New parses this package's pages and each extra file system's templates/ pages against the shared layout.
func New(c *Console, extra ...fs.FS) (*Console, error) {
	c.pages = map[string]*template.Template{}
	c.mux = http.NewServeMux()
	if c.Flush == nil {
		c.Flush = func() {}
	}
	layout, err := fs.ReadFile(templateFS, "templates/layout.html")
	if err != nil {
		return nil, err
	}
	for _, src := range append([]fs.FS{templateFS}, extra...) {
		if err := c.parse(src, string(layout)); err != nil {
			return nil, err
		}
	}
	return c, nil
}

func (c *Console) parse(src fs.FS, layout string) error {
	funcs := template.FuncMap{"asset": web.Asset, "when": when, "ago": ago, "join": strings.Join, "has": has}
	return fs.WalkDir(src, "templates", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".html") || path.Base(p) == "layout.html" {
			return err
		}
		body, err := fs.ReadFile(src, p)
		if err != nil {
			return err
		}
		t, err := template.New("layout").Funcs(funcs).Parse(layout)
		if err == nil {
			_, err = t.New(p).Parse(string(body))
		}
		if err != nil {
			return fmt.Errorf("console: %s: %w", p, err)
		}
		name := strings.TrimSuffix(strings.TrimPrefix(p, "templates/"), ".html")
		if _, dup := c.pages[name]; dup {
			return fmt.Errorf("console: page %s is defined twice", name)
		}
		c.pages[name] = t
		return nil
	})
}

// Register mounts /console behind sign-in, CSRF and same-origin checks.
func (c *Console) Register(mux *http.ServeMux) {
	mux.Handle("/console", c.Sessions.Require("/auth/login", c.mux))
	mux.Handle("/console/", c.Sessions.Require("/auth/login", c.mux))
}

// Handle routes pattern to h when the user holds perm at organisation scope; an empty perm admits any signed-in user.
func (c *Console) Handle(pattern string, perm authz.Perm, h Handler) {
	c.mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
		q, err := c.request(w, r)
		if err != nil {
			http.Error(w, "session error", http.StatusInternalServerError)
			return
		}
		if perm != "" && !q.Grants.Can(perm, "") {
			q.Fail(http.StatusForbidden, "You do not have permission to open this page.")
			return
		}
		if err := h(q); err != nil {
			q.fail(err)
		}
	})
}

func when(t any) string {
	switch v := t.(type) {
	case time.Time:
		if v.IsZero() {
			return "never"
		}
		return v.UTC().Format("2006-01-02 15:04 UTC")
	case *time.Time:
		if v == nil {
			return "never"
		}
		return v.UTC().Format("2006-01-02 15:04 UTC")
	}
	return ""
}

func ago(t any) string {
	var v time.Time
	switch x := t.(type) {
	case time.Time:
		v = x
	case *time.Time:
		if x == nil {
			return "never"
		}
		v = *x
	}
	if v.IsZero() {
		return "never"
	}
	d := time.Since(v)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%d min ago", int(d.Minutes()))
	case d < 48*time.Hour:
		return fmt.Sprintf("%d h ago", int(d.Hours()))
	}
	return fmt.Sprintf("%d days ago", int(d.Hours()/24))
}

func has(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}
