package oauth

import (
	"context"
	"net/http"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ratelimit"
)

// ApprovePage asks a signed-in user to approve a client sign-in.
type ApprovePage struct {
	OrgName    string
	ClientName string
	DeviceName string
	UserEmail  string
	Scopes     []string
	Action     string
	CSRF       string
	Hidden     map[string]string
}

// DevicePage enters or confirms a device code.
type DevicePage struct {
	OrgName    string
	UserCode   string
	ClientName string
	DeviceName string
	UserEmail  string
	CSRF       string
	Error      string
	Confirm    bool
}

// MessagePage is a terminal result screen.
type MessagePage struct {
	Status int
	Title  string
	Body   string
}

// Pages renders the HTML screens of the OAuth flows.
type Pages interface {
	Approve(w http.ResponseWriter, r *http.Request, p ApprovePage)
	Device(w http.ResponseWriter, r *http.Request, p DevicePage)
	Message(w http.ResponseWriter, r *http.Request, p MessagePage)
}

// Server serves /oauth/* and /device.
type Server struct {
	Pool      *pgxpool.Pool
	Issuer    *Issuer
	Sessions  *session.Manager
	Pages     Pages
	Proxies   *httpx.Proxies
	PublicURL string
	OrgName   func(context.Context) string
	Me        func(ctx context.Context, userID string) (discovery.Me, error)
	AccessFor func(ctx context.Context, token string) (userID string, err error)
	RateKey   []byte
	tokenRate *ratelimit.Limiter
	codeRate  *ratelimit.Limiter
}

// LoginPath is where Require sends anonymous browsers.
const LoginPath = "/auth/login"

// Register mounts the endpoints; browser endpoints require a console session.
func (s *Server) Register(mux *http.ServeMux) {
	s.tokenRate = ratelimit.Shared(s.Pool, s.RateKey, "oauth", 5, 30)
	s.codeRate = ratelimit.Shared(s.Pool, s.RateKey, "device", 0.2, 10)
	mux.Handle("GET /oauth/authorize", s.Sessions.Require(LoginPath, http.HandlerFunc(s.authorizeForm)))
	mux.Handle("POST /oauth/authorize", s.Sessions.Require(LoginPath, http.HandlerFunc(s.authorizeDecide)))
	mux.HandleFunc("POST /oauth/token", s.token)
	mux.HandleFunc("POST /oauth/device/code", s.deviceCode)
	mux.Handle("GET /device", s.Sessions.Require(LoginPath, http.HandlerFunc(s.deviceForm)))
	mux.Handle("POST /device", s.Sessions.Require(LoginPath, http.HandlerFunc(s.deviceLookup)))
	mux.Handle("POST /device/decide", s.Sessions.Require(LoginPath, http.HandlerFunc(s.deviceDecide)))
	mux.HandleFunc("POST /oauth/revoke", s.revoke)
	mux.HandleFunc("GET /oauth/userinfo", s.userinfo)
}

func (s *Server) org(ctx context.Context) string {
	if s.OrgName == nil {
		return "TensorFold Starlight"
	}
	return s.OrgName(ctx)
}

func (s *Server) record(r *http.Request, action, outcome string, details map[string]any) {
	e := audit.FromRequest(r, s.Proxies, nil)
	if _, u, ok := session.From(r.Context()); ok {
		e.ActorType, e.ActorID, e.ActorLabel = "user", u.ID, u.Email
	}
	e.Action, e.Outcome, e.Details = action, outcome, details
	_ = audit.Log(context.WithoutCancel(r.Context()), s.Pool, e)
}

// clientKey is the request's client network as the token limit counts it.
func (s *Server) clientKey(r *http.Request) string {
	return ratelimit.Client(s.Proxies.ClientIP(r))
}

type oauthError struct {
	status int
	code   string
	desc   string
}

func writeOAuthError(w http.ResponseWriter, e oauthError) {
	w.Header().Set("Pragma", "no-cache")
	httpx.JSON(w, e.status, map[string]string{"error": e.code, "error_description": e.desc})
}

func invalidRequest(desc string) oauthError {
	return oauthError{http.StatusBadRequest, "invalid_request", desc}
}

var (
	errInvalidClient = oauthError{http.StatusUnauthorized, "invalid_client", "unknown client"}
	errInvalidGrant  = oauthError{http.StatusBadRequest, "invalid_grant", "the grant is invalid, expired or already used"}
	errSlowDown      = oauthError{http.StatusTooManyRequests, "slow_down", "too many requests"}
)
