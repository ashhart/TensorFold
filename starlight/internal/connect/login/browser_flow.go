package login

import (
	"context"
	"crypto/subtle"
	"errors"
	"fmt"
	"html"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/oauth"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

// Options tune a sign-in; zero values take Connect's defaults.
type Options struct {
	Client     *http.Client
	Open       func(string) error
	Out        io.Writer
	Timeout    time.Duration
	DeviceName string
	unit       time.Duration
}

func (o *Options) defaults() {
	if o.Client == nil {
		o.Client = &http.Client{Timeout: 30 * time.Second}
	}
	if o.Open == nil {
		o.Open = OpenBrowser
	}
	if o.Out == nil {
		o.Out = io.Discard
	}
	if o.Timeout <= 0 {
		o.Timeout = 5 * time.Minute
	}
	if o.unit <= 0 {
		o.unit = time.Second
	}
}

type callback struct {
	code string
	err  error
}

// Browser runs authorization code with PKCE through a one-shot listener on a loopback port.
func Browser(ctx context.Context, doc discovery.Document, o Options) (discovery.TokenResponse, error) {
	o.defaults()
	ctx, cancel := context.WithTimeout(ctx, o.Timeout)
	defer cancel()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return discovery.TokenResponse{}, fmt.Errorf("start sign-in listener: %w", err)
	}
	redirect := fmt.Sprintf("http://127.0.0.1:%d/callback", ln.Addr().(*net.TCPAddr).Port)
	verifier, challenge := oauth.PKCE()
	state := oauth.State()
	results := make(chan callback, 1)
	srv := &http.Server{Handler: callbackHandler(state, results), ReadHeaderTimeout: 10 * time.Second}
	go srv.Serve(ln)
	defer srv.Close()

	q := url.Values{
		"response_type":         {"code"},
		"client_id":             {discovery.ClientConnect},
		"redirect_uri":          {redirect},
		"scope":                 {oauth.Scope},
		"state":                 {state},
		"code_challenge":        {challenge},
		"code_challenge_method": {"S256"},
	}
	if o.DeviceName != "" {
		q.Set("device_name", o.DeviceName)
	}
	authURL := doc.AuthorizationEndpoint + "?" + q.Encode()
	fmt.Fprintf(o.Out, "Opening your browser to sign in. If it does not open, visit:\n\n  %s\n\n", authURL)
	if err := o.Open(authURL); err != nil {
		fmt.Fprintf(o.Out, "Could not open a browser (%v); open the address above yourself.\n", err)
	}

	var res callback
	select {
	case res = <-results:
	case <-ctx.Done():
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return discovery.TokenResponse{}, errors.New("sign-in timed out; run login again")
		}
		return discovery.TokenResponse{}, ctx.Err()
	}
	if res.err != nil {
		return discovery.TokenResponse{}, res.err
	}
	return oauth.Token(ctx, o.Client, doc.TokenEndpoint, url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {res.code},
		"redirect_uri":  {redirect},
		"client_id":     {discovery.ClientConnect},
		"code_verifier": {verifier},
	})
}

// callbackHandler accepts exactly one redirect carrying the expected state and ignores everything else.
func callbackHandler(state string, results chan<- callback) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/callback" || r.Method != http.MethodGet {
			http.NotFound(w, r)
			return
		}
		q := r.URL.Query()
		if subtle.ConstantTimeCompare([]byte(q.Get("state")), []byte(state)) != 1 {
			page(w, http.StatusBadRequest, "This sign-in link does not belong to the sign-in TensorFold Connect started. Close it and try again.")
			return
		}
		if e := q.Get("error"); e != "" {
			page(w, http.StatusBadRequest, "Sign-in did not complete: "+e+". Return to TensorFold Connect and try again.")
			send(results, callback{err: &oauth.Error{Code: e, Description: q.Get("error_description")}})
			return
		}
		code := q.Get("code")
		if code == "" {
			page(w, http.StatusBadRequest, "The sign-in reply carried no authorization code.")
			send(results, callback{err: errors.New("authorization reply carried no code")})
			return
		}
		page(w, http.StatusOK, "You are signed in to TensorFold Connect. You can close this window.")
		send(results, callback{code: code})
	})
}

func send(ch chan<- callback, c callback) {
	select {
	case ch <- c:
	default:
	}
}

func page(w http.ResponseWriter, status int, msg string) {
	h := w.Header()
	h.Set("Content-Type", "text/html; charset=utf-8")
	h.Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
	h.Set("Referrer-Policy", "no-referrer")
	h.Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	fmt.Fprintf(w, `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">`+
		`<title>TensorFold Connect</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#080B17;color:#fff;`+
		`font:16px/1.5 system-ui,sans-serif}main{max-width:28rem;padding:2rem;text-align:center}h1{font-size:1.25rem;`+
		`background:linear-gradient(90deg,#FF7A5C,#8B4DFF,#3D6BFF);-webkit-background-clip:text;color:transparent}</style></head>`+
		`<body><main><h1>TensorFold Connect</h1><p>%s</p></main></body></html>`, html.EscapeString(msg))
}
