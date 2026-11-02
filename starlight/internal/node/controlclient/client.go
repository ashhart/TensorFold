// Package controlclient speaks the node side of Starlight's control protocol.
package controlclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

const maxResponse = 8 << 20

// StatusError is a non-2xx answer from Starlight.
type StatusError struct {
	Status  int
	Code    string
	Message string
}

func (e *StatusError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("starlight answered %d %s: %s", e.Status, e.Code, e.Message)
	}
	return fmt.Sprintf("starlight answered %d", e.Status)
}

// Revoked reports whether Starlight refused the node's identity, so retrying cannot help.
func Revoked(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && (se.Status == http.StatusUnauthorized || se.Status == http.StatusForbidden)
}

// Client posts JSON to one Starlight base URL.
type Client struct {
	HTTP *http.Client
	Base *url.URL
}

// Enroll exchanges the enrolment token and CSR for an identity.
func (c *Client) Enroll(ctx context.Context, req wire.EnrollRequest) (wire.EnrollResponse, error) {
	var out wire.EnrollResponse
	return out, c.post(ctx, wire.PathEnroll, req, &out)
}

// Sync reports state and waits for the desired state.
func (c *Client) Sync(ctx context.Context, req wire.SyncRequest) (wire.SyncResponse, error) {
	var out wire.SyncResponse
	return out, c.post(ctx, wire.PathSync, req, &out)
}

// Stats posts one live stats report; Starlight answers 204.
func (c *Client) Stats(ctx context.Context, rep wire.StatsReport) error {
	return c.post(ctx, wire.PathStats, rep, nil)
}

// Renew asks for a certificate for a fresh key.
func (c *Client) Renew(ctx context.Context, req wire.RenewRequest) (wire.RenewResponse, error) {
	var out wire.RenewResponse
	return out, c.post(ctx, wire.PathRenew, req, &out)
}

func (c *Client) post(ctx context.Context, path string, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return err
	}
	u := *c.Base
	u.Path = u.Path + path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponse+1))
	if err != nil {
		return err
	}
	if len(raw) > maxResponse {
		return errors.New("starlight response too large")
	}
	if resp.StatusCode/100 != 2 {
		return decodeError(resp.StatusCode, raw)
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("starlight response: %w", err)
	}
	return nil
}

func decodeError(status int, raw []byte) error {
	var env struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
	}
	se := &StatusError{Status: status}
	if json.Unmarshal(raw, &env) != nil {
		return se
	}
	var nested struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if json.Unmarshal(env.Error, &nested) == nil && (nested.Code != "" || nested.Message != "") {
		se.Code, se.Message = nested.Code, nested.Message
		return se
	}
	var flat string
	if json.Unmarshal(env.Error, &flat) == nil {
		se.Code, se.Message = flat, env.Message
	}
	return se
}
