// Package httpx holds HTTP plumbing shared by every Starlight listener.
package httpx

import (
	"encoding/json"
	"net/http"
	"strconv"
)

// Dialect selects the error envelope a client expects.
type Dialect int

const (
	OpenAI Dialect = iota
	Anthropic
	Plain
)

// Error is a client-facing failure with a stable machine-readable code.
type Error struct {
	Status     int
	Code       string
	Message    string
	Param      string
	RetryAfter int
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// Errorf builds an Error.
func Errorf(status int, code, msg string) *Error {
	return &Error{Status: status, Code: code, Message: msg}
}

// Write sends err in the dialect's envelope; unknown errors become an opaque 500.
func Write(w http.ResponseWriter, d Dialect, err error) {
	e, ok := err.(*Error)
	if !ok {
		e = &Error{Status: http.StatusInternalServerError, Code: "internal_error", Message: "internal error"}
	}
	if e.RetryAfter > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(e.RetryAfter))
	}
	if e.Status == http.StatusUnauthorized {
		w.Header().Set("WWW-Authenticate", `Bearer realm="starlight"`)
	}
	switch d {
	case Anthropic:
		JSON(w, e.Status, map[string]any{"type": "error", "error": map[string]any{"type": anthropicType(e.Status), "message": e.Message}})
	case Plain:
		JSON(w, e.Status, map[string]any{"error": e.Code, "message": e.Message})
	default:
		body := map[string]any{"message": e.Message, "type": openAIType(e.Status), "code": e.Code, "param": nil}
		if e.Param != "" {
			body["param"] = e.Param
		}
		JSON(w, e.Status, map[string]any{"error": body})
	}
}

func openAIType(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status >= 500:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}

func anthropicType(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == http.StatusServiceUnavailable:
		return "overloaded_error"
	case status >= 500:
		return "api_error"
	default:
		return "invalid_request_error"
	}
}

// JSON writes v with the status and no-store caching.
func JSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// Common errors.
var (
	ErrUnauthenticated = Errorf(http.StatusUnauthorized, "invalid_api_key", "missing or invalid credentials")
	ErrForbidden       = Errorf(http.StatusForbidden, "forbidden", "not permitted")
	ErrNotFound        = Errorf(http.StatusNotFound, "not_found", "not found")
	ErrTooLarge        = Errorf(http.StatusRequestEntityTooLarge, "request_too_large", "request body is too large")
)
