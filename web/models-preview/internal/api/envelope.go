// Package api is the JSON transport: each Endpoint is served over HTTP and called in-process by the UI.
package api

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

// Envelope is the shape of every JSON response. Details is only set for validation errors (field -> message).
type Envelope[Value any] struct {
	Data    *Value            `json:"data"`
	Error   *string           `json:"error"`
	Details map[string]string `json:"details,omitempty"`
}

// WriteData writes a success envelope.
func WriteData[Value any](writer http.ResponseWriter, status int, data Value) {
	write(writer, status, Envelope[Value]{Data: &data})
}

// WriteError writes an error envelope.
func WriteError(writer http.ResponseWriter, status int, msg string, details map[string]string) {
	write(writer, status, Envelope[struct{}]{Error: &msg, Details: details})
}

// writeErr maps validation and service errors to a status and a safe message; anything else is a 500.
func writeErr(writer http.ResponseWriter, request *http.Request, err error) {
	var verr *validate.Error
	var serr *services.Error
	switch {
	case errors.As(err, &verr):
		WriteError(writer, http.StatusBadRequest, "Invalid request data", verr.Fields)
	case errors.As(err, &serr) && serr.Kind != services.KindInternal:
		WriteError(writer, statusFor(serr.Kind), serr.Message, nil)
	default:
		slog.ErrorContext(request.Context(), "request failed", "method", request.Method, "path", request.URL.Path, "err", err)
		WriteError(writer, http.StatusInternalServerError, "Internal server error", nil)
	}
}

func statusFor(kind services.Kind) int {
	switch kind {
	case services.KindInvalid:
		return http.StatusBadRequest
	case services.KindNotFound:
		return http.StatusNotFound
	case services.KindUnavailable:
		return http.StatusServiceUnavailable
	default:
		return http.StatusInternalServerError
	}
}

func write(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}
