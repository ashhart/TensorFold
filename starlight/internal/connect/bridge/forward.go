package bridge

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/textproto"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/connect/appkeys"
	"github.com/ashhart/TensorFold/starlight/internal/connect/tokens"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

// dropped request headers: hop-by-hop, the client's own credentials, and anything Connect sets itself.
var dropRequest = map[string]bool{
	"Authorization": true, "X-Api-Key": true, "Cookie": true, "Proxy-Authorization": true, "Proxy-Connection": true,
	"Connection": true, "Keep-Alive": true, "Te": true, "Trailer": true, "Transfer-Encoding": true, "Upgrade": true,
	"Host": true, "Origin": true, "Referer": true, "Accept-Encoding": true, "Content-Length": true, "X-Starlight-Client": true,
	"Forwarded": true, "X-Forwarded-For": true, "X-Forwarded-Host": true, "X-Forwarded-Proto": true, "X-Real-Ip": true,
}

var dropResponse = map[string]bool{
	"Connection": true, "Keep-Alive": true, "Proxy-Authenticate": true, "Te": true, "Trailer": true, "Transfer-Encoding": true,
	"Upgrade": true, "Set-Cookie": true, "Content-Length": true, "Strict-Transport-Security": true,
}

// body is the request body, buffered when small enough to resend after a token refresh.
type body struct {
	buf    []byte
	stream io.Reader
}

func (b body) replayable() bool { return b.stream == nil }

func (b body) reader() io.Reader {
	if b.stream != nil {
		return b.stream
	}
	return bytes.NewReader(b.buf)
}

func readBody(r *http.Request, limit int64) (body, error) {
	if r.Body == nil || r.Body == http.NoBody {
		return body{}, nil
	}
	if r.ContentLength > limit {
		return body{stream: r.Body}, nil
	}
	buf, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return body{}, err
	}
	if int64(len(buf)) > limit {
		return body{stream: io.MultiReader(bytes.NewReader(buf), r.Body)}, nil
	}
	return body{buf: buf}, nil
}

func (s *Server) forward(w http.ResponseWriter, r *http.Request, app appkeys.App, d httpx.Dialect) {
	ctx := r.Context()
	b, err := readBody(r, s.cfg.MaxReplayBody)
	if err != nil {
		httpx.Write(w, d, httpx.Errorf(http.StatusBadRequest, "invalid_body", "could not read the request body"))
		return
	}
	tok, err := s.cfg.Tokens.Access(ctx)
	if err != nil {
		s.tokenError(w, d, err)
		return
	}
	resp, err := s.send(ctx, r, b, tok, app)
	if err == nil && resp.StatusCode == http.StatusUnauthorized && b.replayable() {
		resp.Body.Close()
		tok, err = s.cfg.Tokens.Refresh(ctx, tok)
		if err != nil {
			s.tokenError(w, d, err)
			return
		}
		resp, err = s.send(ctx, r, b, tok, app)
	}
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		httpx.Write(w, d, httpx.Errorf(http.StatusBadGateway, "starlight_unreachable", "TensorFold Connect could not reach Starlight"))
		return
	}
	defer resp.Body.Close()
	h := w.Header()
	for k, vs := range resp.Header {
		if !dropResponse[textproto.CanonicalMIMEHeaderKey(k)] {
			h[k] = append([]string(nil), vs...)
		}
	}
	w.WriteHeader(resp.StatusCode)
	stream(w, resp.Body)
}

func (s *Server) send(ctx context.Context, r *http.Request, b body, token string, app appkeys.App) (*http.Response, error) {
	u := *s.base
	u.Path = strings.TrimRight(s.base.Path, "/") + r.URL.Path
	u.RawPath = ""
	u.RawQuery = r.URL.RawQuery
	req, err := http.NewRequestWithContext(ctx, r.Method, u.String(), b.reader())
	if err != nil {
		return nil, err
	}
	if b.replayable() {
		req.ContentLength = int64(len(b.buf))
	}
	skip := map[string]bool{}
	for _, c := range r.Header.Values("Connection") {
		for _, name := range strings.Split(c, ",") {
			skip[textproto.CanonicalMIMEHeaderKey(strings.TrimSpace(name))] = true
		}
	}
	for k, vs := range r.Header {
		ck := textproto.CanonicalMIMEHeaderKey(k)
		if !dropRequest[ck] && !skip[ck] {
			req.Header[ck] = append([]string(nil), vs...)
		}
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("X-Starlight-Client", "connect/"+app.Name)
	if req.Header.Get("X-Starlight-Project") == "" && s.cfg.Project != "" {
		req.Header.Set("X-Starlight-Project", s.cfg.Project)
	}
	return s.cfg.HTTP.Do(req)
}

// stream copies the reply and flushes every chunk so server-sent events reach the client as they arrive.
func stream(w http.ResponseWriter, src io.Reader) {
	rc := http.NewResponseController(w)
	buf := make([]byte, 32<<10)
	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			_ = rc.Flush()
		}
		if err != nil {
			return
		}
	}
}

func (s *Server) tokenError(w http.ResponseWriter, d httpx.Dialect, err error) {
	if errors.Is(err, tokens.ErrSignedOut) {
		httpx.Write(w, d, httpx.Errorf(http.StatusUnauthorized, "signed_out", tokens.ErrSignedOut.Error()))
		return
	}
	httpx.Write(w, d, httpx.Errorf(http.StatusBadGateway, "token_refresh_failed", "TensorFold Connect could not refresh its Starlight sign-in"))
}
