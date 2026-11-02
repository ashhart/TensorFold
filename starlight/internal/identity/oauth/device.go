package oauth

import (
	"net/http"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

const deviceLifetime = 10 * time.Minute

func (s *Server) deviceCode(w http.ResponseWriter, r *http.Request) {
	if ok, _ := s.tokenRate.Take(r.Context(), s.clientKey(r)); !ok {
		writeOAuthError(w, errSlowDown)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 16<<10)
	if err := r.ParseForm(); err != nil {
		writeOAuthError(w, invalidRequest("form body required"))
		return
	}
	client, ok := LookupClient(r.PostForm.Get("client_id"))
	if !ok || !client.DeviceGrant {
		writeOAuthError(w, errInvalidClient)
		return
	}
	scope, ok := client.NarrowScope(r.PostForm.Get("scope"))
	if !ok {
		writeOAuthError(w, oauthError{http.StatusBadRequest, "invalid_scope", "scope is not available to this client"})
		return
	}
	device, hash := secret.Generate(secret.Device)
	var userCode string
	for attempt := 0; attempt < 5; attempt++ {
		userCode = secret.UserCode()
		_, err := s.Pool.Exec(r.Context(), `INSERT INTO device_grants (id, device_code_hash, user_code, client_id, scope, device_name, expires_at)
			VALUES ($1,$2,$3,$4,$5,$6,$7)`, ids.New(), hash, userCode, client.ID, scope, DeviceName(r.PostForm.Get("device_name")),
			time.Now().Add(deviceLifetime))
		if err == nil {
			break
		}
		if !db.IsUnique(err, "") || attempt == 4 {
			writeOAuthError(w, oauthError{http.StatusInternalServerError, "server_error", "try again"})
			return
		}
	}
	verify := strings.TrimRight(s.PublicURL, "/") + "/device"
	httpx.JSON(w, http.StatusOK, discovery.DeviceResponse{DeviceCode: device, UserCode: userCode, VerificationURI: verify,
		VerificationURIComplete: verify + "?user_code=" + userCode, ExpiresIn: int(deviceLifetime.Seconds()), Interval: 5})
}

func (s *Server) deviceForm(w http.ResponseWriter, r *http.Request) {
	sess, u, _ := session.From(r.Context())
	s.Pages.Device(w, r, DevicePage{OrgName: s.org(r.Context()), UserCode: secret.NormalizeUserCode(r.URL.Query().Get("user_code")),
		UserEmail: u.Email, CSRF: sess.CSRF()})
}

type pendingDevice struct {
	id, client, device string
}

func (s *Server) findPending(r *http.Request, code string) (pendingDevice, bool) {
	var p pendingDevice
	err := s.Pool.QueryRow(r.Context(), `SELECT id, client_id, device_name FROM device_grants
		WHERE user_code = $1 AND status = 'pending' AND expires_at > now()`, code).Scan(&p.id, &p.client, &p.device)
	return p, err == nil
}

func (s *Server) deviceLookup(w http.ResponseWriter, r *http.Request) {
	sess, u, _ := session.From(r.Context())
	page := DevicePage{OrgName: s.org(r.Context()), UserEmail: u.Email, CSRF: sess.CSRF()}
	if ok, _ := s.codeRate.Take(r.Context(), u.ID); !ok {
		page.Error = "Too many attempts. Wait a minute and try again."
		s.Pages.Device(w, r, page)
		return
	}
	code := secret.NormalizeUserCode(r.PostFormValue("user_code"))
	p, ok := s.findPending(r, code)
	if code == "" || !ok {
		page.Error = "That code is not valid or has expired."
		s.Pages.Device(w, r, page)
		return
	}
	c, _ := LookupClient(p.client)
	page.UserCode, page.ClientName, page.DeviceName, page.Confirm = code, c.Name, p.device, true
	s.Pages.Device(w, r, page)
}

func (s *Server) deviceDecide(w http.ResponseWriter, r *http.Request) {
	_, u, _ := session.From(r.Context())
	if ok, _ := s.codeRate.Take(r.Context(), u.ID); !ok {
		s.Pages.Message(w, r, MessagePage{Status: http.StatusTooManyRequests, Title: "Too many attempts", Body: "Wait a minute and try again."})
		return
	}
	code := secret.NormalizeUserCode(r.PostFormValue("user_code"))
	p, ok := s.findPending(r, code)
	if code == "" || !ok {
		s.Pages.Message(w, r, MessagePage{Status: http.StatusBadRequest, Title: "Code expired", Body: "Start the sign-in again on your device."})
		return
	}
	status, title, body := "denied", "Request denied", "The device was not signed in."
	if r.PostFormValue("decision") == "allow" {
		status, title, body = "approved", "Device signed in", "You can return to your device."
	}
	tag, err := s.Pool.Exec(r.Context(), "UPDATE device_grants SET status = $2, user_id = $3 WHERE id = $1 AND status = 'pending'", p.id, status, u.ID)
	if err != nil || tag.RowsAffected() != 1 {
		s.Pages.Message(w, r, MessagePage{Status: http.StatusConflict, Title: "Code already used", Body: "Start the sign-in again on your device."})
		return
	}
	s.record(r, "oauth.device", status, map[string]any{"client": p.client, "device": p.device})
	s.Pages.Message(w, r, MessagePage{Status: http.StatusOK, Title: title, Body: body})
}
