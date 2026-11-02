package console

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/principal"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// Request is one console request with the signed-in user's identity and grants.
type Request struct {
	W       http.ResponseWriter
	R       *http.Request
	User    directory.User
	Session session.Session
	Grants  authz.Grants
	Groups  []string
	c       *Console
	flash   string
}

const flashCookie = "starlight_flash"

func (c *Console) request(w http.ResponseWriter, r *http.Request) (*Request, error) {
	s, u, ok := session.From(r.Context())
	if !ok {
		return nil, errors.New("console: no session in context")
	}
	groups, err := directory.GroupIDs(r.Context(), c.Pool, u.ID)
	if err != nil {
		return nil, err
	}
	g, err := authz.LoadUser(r.Context(), c.Pool, u.ID, groups)
	if err != nil {
		return nil, err
	}
	q := &Request{W: w, R: r, User: u, Session: s, Grants: g, Groups: groups, c: c}
	if f, err := r.Cookie(flashCookie); err == nil {
		q.flash, _ = url.QueryUnescape(f.Value)
		http.SetCookie(w, &http.Cookie{Name: flashCookie, Path: "/console", MaxAge: -1, HttpOnly: true, SameSite: http.SameSiteStrictMode})
	}
	return q, nil
}

// Ctx is the request context.
func (q *Request) Ctx() context.Context { return q.R.Context() }

// Can checks a permission, at project scope when project is set.
func (q *Request) Can(p authz.Perm, project string) bool { return q.Grants.Can(p, project) }

// Principal is the acting user for audit records.
func (q *Request) Principal() *principal.Principal {
	return &principal.Principal{Type: principal.User, ID: q.User.ID, UserID: q.User.ID, Email: q.User.Email, Via: principal.ViaSession}
}

// Audit builds an event for this request; callers set Action, Target and Details, then append it in their transaction.
func (q *Request) Audit(action, targetType, targetID string, details map[string]any) audit.Event {
	e := audit.FromRequest(q.R, q.c.Proxies, q.Principal())
	e.Action, e.TargetType, e.TargetID, e.Details = action, targetType, targetID, details
	return e
}

// Form returns a trimmed form value.
func (q *Request) Form(k string) string { return strings.TrimSpace(q.R.PostFormValue(k)) }

// Bool reads a checkbox.
func (q *Request) Bool(k string) bool {
	v := q.R.PostFormValue(k)
	return v == "on" || v == "true" || v == "1"
}

// Int reads an integer field, or def when it is empty.
func (q *Request) Int(k string, def int) (int, error) {
	v := q.Form(k)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, Invalid(k + " must be a whole number")
	}
	return n, nil
}

// Lines splits a textarea into non-empty trimmed lines.
func (q *Request) Lines(k string) []string {
	var out []string
	for _, l := range strings.Split(q.R.PostFormValue(k), "\n") {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return out
}

// view is the data every console page receives.
type view struct {
	Title   string
	Org     string
	Version string
	User    directory.User
	CSRF    string
	Nav     []navSection
	Active  string
	Flash   string
	Error   string
	Page    any
}

// Render draws a page inside the console layout.
func (q *Request) Render(page, title string, data any) {
	q.render(http.StatusOK, page, title, data, "")
}

func (q *Request) render(status int, page, title string, data any, errMsg string) {
	t, ok := q.c.pages[page]
	if !ok {
		t = q.c.pages["message"]
	}
	v := view{Title: title, Org: q.c.Org(q.Ctx()), Version: q.c.Version, User: q.User, CSRF: q.Session.CSRF(), Nav: q.nav(),
		Active: q.R.URL.Path, Flash: q.flash, Error: errMsg, Page: data}
	var buf bytes.Buffer
	if err := t.ExecuteTemplate(&buf, "layout", v); err != nil {
		slog.Error("console render", "page", page, "err", err)
		http.Error(q.W, "page failed to render", http.StatusInternalServerError)
		return
	}
	h := q.W.Header()
	h.Set("Content-Type", "text/html; charset=utf-8")
	h.Set("Cache-Control", "no-store")
	h.Set("Content-Security-Policy", web.ConsoleCSP)
	q.W.WriteHeader(status)
	_, _ = q.W.Write(buf.Bytes())
}

// Redirect sends the browser to path with a one-time notice.
func (q *Request) Redirect(path, notice string) error {
	if notice != "" {
		http.SetCookie(q.W, &http.Cookie{Name: flashCookie, Value: url.QueryEscape(notice), Path: "/console", MaxAge: 60, HttpOnly: true,
			Secure: q.R.TLS != nil, SameSite: http.SameSiteStrictMode})
	}
	http.Redirect(q.W, q.R, path, http.StatusSeeOther)
	return nil
}

// Fail renders an error page.
func (q *Request) Fail(status int, msg string) {
	q.render(status, "message", http.StatusText(status), messagePage{Heading: http.StatusText(status), Body: msg}, msg)
}

// Invalid is a user-correctable form error.
type Invalid string

func (e Invalid) Error() string { return string(e) }

// Rerender is an Invalid error that redraws the form with the user's input.
type Rerender struct {
	Page, Title string
	Data        any
	Msg         string
}

func (e *Rerender) Error() string { return e.Msg }

func (q *Request) fail(err error) {
	var inv Invalid
	var rr *Rerender
	var he *httpx.Error
	switch {
	case errors.As(err, &rr):
		q.render(http.StatusUnprocessableEntity, rr.Page, rr.Title, rr.Data, rr.Msg)
	case errors.As(err, &inv):
		q.Fail(http.StatusUnprocessableEntity, string(inv))
	case errors.As(err, &he):
		q.Fail(he.Status, he.Message)
	case errors.Is(err, directory.ErrNotFound):
		q.Fail(http.StatusNotFound, "That record does not exist.")
	default:
		slog.Error("console action failed", "path", q.R.URL.Path, "err", err)
		q.Fail(http.StatusInternalServerError, "Something went wrong. The error was logged with this page's reference.")
	}
}

type messagePage struct {
	Heading string
	Body    string
}
