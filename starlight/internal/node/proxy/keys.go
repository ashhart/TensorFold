package proxy

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
)

// keyFor picks the least-privileged engine key: the full key only where the engine's scope rules demand it.
func keyFor(method, path string, body []byte, keys engine.Keys) string {
	switch {
	case (method == http.MethodGet || method == http.MethodDelete) && strings.HasPrefix(path, "/v1/responses/"):
		return keys.Responses
	case method == http.MethodPost && path == "/v1/responses" && storesOrContinues(body):
		return keys.Responses
	default:
		return keys.Inference
	}
}

// storesOrContinues reports whether a Responses body keeps state: store absent or true, or a previous_response_id.
func storesOrContinues(body []byte) bool {
	var req struct {
		Store    *bool  `json:"store"`
		Previous string `json:"previous_response_id"`
	}
	if json.Unmarshal(body, &req) != nil {
		return false
	}
	return req.Store == nil || *req.Store || req.Previous != ""
}
