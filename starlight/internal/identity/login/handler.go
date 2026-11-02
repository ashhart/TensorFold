// Package login serves Starlight's sign-in endpoints: SSO starts and callbacks, logout, owner setup and break-glass.
package login

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/enrich"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	samlsp "github.com/ashhart/TensorFold/starlight/internal/identity/saml"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// Config wires the login handlers to Starlight's shared services.
type Config struct {
	PublicURL       string
	Pool            *pgxpool.Pool
	Keyring         *keyring.Keyring
	Sessions        *session.Manager
	Proxies         *httpx.Proxies
	Web             *web.Renderer
	HTTP            *http.Client
	Entra           *enrich.Entra
	Google          *enrich.Google
	Now             func() time.Time
	StartsPerMinute int
	DefaultReturn   string
	RateKey         []byte
}

// Handler serves /auth/*.
type Handler struct {
	cfg     Config
	secure  bool
	conns   *connections.Store
	rp      *oidc.RP
	samls   *samlsp.Factory
	bearer  *oidc.Bearer
	limiter *limiter
	// samlLimiter bounds ACS and logout posts; a sign-in and its sign-out each post once, so it allows twice the starts.
	samlLimiter *limiter
}

// New validates cfg and builds the handler.
func New(cfg Config) (*Handler, error) {
	if cfg.Pool == nil || cfg.Keyring == nil || cfg.Sessions == nil || cfg.Web == nil {
		return nil, errors.New("login: pool, keyring, sessions and web renderer are required")
	}
	cfg.PublicURL = strings.TrimRight(cfg.PublicURL, "/")
	if !strings.HasPrefix(cfg.PublicURL, "https://") && !strings.HasPrefix(cfg.PublicURL, "http://") {
		return nil, errors.New("login: public URL must be absolute")
	}
	if cfg.HTTP == nil {
		cfg.HTTP = &http.Client{Timeout: 15 * time.Second}
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Entra == nil {
		cfg.Entra = enrich.NewEntra(cfg.HTTP)
	}
	if cfg.Google == nil {
		cfg.Google = enrich.NewGoogle(cfg.HTTP)
	}
	if cfg.StartsPerMinute <= 0 {
		cfg.StartsPerMinute = 30
	}
	if cfg.DefaultReturn == "" {
		cfg.DefaultReturn = "/console"
	}
	conns := connections.NewStore(cfg.Pool, cfg.Keyring)
	rp := oidc.NewRP(cfg.HTTP, cfg.Now)
	return &Handler{
		cfg: cfg, secure: strings.HasPrefix(cfg.PublicURL, "https://"), conns: conns, rp: rp,
		samls: samlsp.NewFactory(cfg.Pool, cfg.HTTP, cfg.PublicURL, cfg.Now), bearer: oidc.NewBearer(rp, conns),
		limiter: newLimiter(cfg, "login", cfg.StartsPerMinute), samlLimiter: newLimiter(cfg, "saml", 2*cfg.StartsPerMinute),
	}, nil
}

// Register mounts every sign-in route on mux; the server mounts web.StaticHandler at /static/ separately.
func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /auth/login", h.loginPage)
	mux.HandleFunc("POST /auth/login", h.loginEmail)
	mux.HandleFunc("GET /auth/{slug}/start", h.start)
	mux.HandleFunc("GET /auth/{slug}/callback", h.oidcCallback)
	mux.HandleFunc("GET /auth/{slug}/saml/metadata", h.samlMetadata)
	mux.HandleFunc("POST /auth/{slug}/saml/acs", h.samlACS)
	mux.HandleFunc("GET /auth/{slug}/saml/slo", h.samlSLO)
	mux.HandleFunc("POST /auth/{slug}/saml/slo", h.samlSLO)
	mux.HandleFunc("POST /auth/{slug}/backchannel-logout", h.backchannel)
	mux.HandleFunc("POST /auth/logout", h.logout)
	mux.HandleFunc("GET /auth/setup", h.setupPage)
	mux.HandleFunc("POST /auth/setup", h.setupPost)
	mux.HandleFunc("GET /auth/recover", h.recoverPage)
	mux.HandleFunc("POST /auth/recover", h.recoverPost)
}

// Connections exposes the connection store for the console.
func (h *Handler) Connections() *connections.Store { return h.conns }

// Forget drops cached discovery and SAML state after a connection is edited.
func (h *Handler) Forget(connectionID string) {
	h.rp.Forget(connectionID)
	h.samls.Forget(connectionID)
}

// BearerUser authenticates an IdP-issued JWT access token and returns the Starlight user already linked to it.
func (h *Handler) BearerUser(ctx context.Context, raw string) (directory.User, connections.Connection, error) {
	conn, res, err := h.bearer.Verify(ctx, raw)
	if err != nil {
		return directory.User{}, conn, err
	}
	id, err := mapping.FromClaims(conn, res.Subject, res.Claims)
	if err != nil {
		return directory.User{}, conn, err
	}
	u, err := mapping.Lookup(ctx, h.cfg.Pool, conn, id)
	return u, conn, err
}

func (h *Handler) ip(r *http.Request) string {
	if a := h.cfg.Proxies.ClientIP(r); a.IsValid() {
		return a.String()
	}
	return ""
}

func (h *Handler) redirectURI(slug string) string {
	return h.cfg.PublicURL + "/auth/" + slug + "/callback"
}

func (h *Handler) record(r *http.Request, action, outcome string, conn *connections.Connection, user *directory.User, details map[string]any) {
	if err := audit.Log(context.WithoutCancel(r.Context()), h.cfg.Pool, h.event(r, action, outcome, conn, user, details)); err != nil {
		httpxLog(r, "audit write failed", err)
	}
}

func (h *Handler) event(r *http.Request, action, outcome string, conn *connections.Connection, user *directory.User, details map[string]any) audit.Event {
	e := audit.FromRequest(r, h.cfg.Proxies, nil)
	if user != nil {
		e.ActorType, e.ActorID, e.ActorLabel = "user", user.ID, user.Email
	} else {
		e.ActorType, e.ActorID = "anonymous", h.ip(r)
	}
	e.Action, e.Outcome, e.Details = action, outcome, details
	if conn != nil {
		e.TargetType, e.TargetID = "idp_connection", conn.ID
		if e.Details == nil {
			e.Details = map[string]any{}
		}
		e.Details["connection"] = conn.Slug
	}
	return e
}

func (h *Handler) fail(w http.ResponseWriter, r *http.Request, status int, heading, body string) {
	h.cfg.Web.Message(w, r, status, "error", heading, body, web.Link{Href: "/auth/login", Label: "Back to sign-in"})
}
