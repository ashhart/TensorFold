// Package starlightapi calls the Starlight endpoints Connect reads: the signed-in user and the model list.
package starlightapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"

	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

// TokenSource supplies access tokens and replaces one Starlight rejected.
type TokenSource interface {
	Access(ctx context.Context) (string, error)
	Refresh(ctx context.Context, stale string) (string, error)
}

// Client is an authenticated Starlight API client.
type Client struct {
	HTTP   *http.Client
	Doc    discovery.Document
	Tokens TokenSource
}

// Model is one entry of /v1/models as the signed-in user may use it.
type Model struct {
	ID      string `json:"id"`
	OwnedBy string `json:"owned_by"`
}

// Error is a non-2xx Starlight reply.
type Error struct {
	Status  int
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("starlight answered %d: %s", e.Status, e.Message) }

// Me returns the signed-in user, their organisation and projects.
func (c *Client) Me(ctx context.Context) (discovery.Me, error) {
	var me discovery.Me
	err := c.get(ctx, c.Doc.UserinfoEndpoint, "", &me)
	return me, err
}

// Models lists the models the user may call, in project when one is named.
func (c *Client) Models(ctx context.Context, project string) ([]Model, error) {
	var list struct {
		Data []Model `json:"data"`
	}
	if err := c.get(ctx, c.Doc.APIBase+"/models", project, &list); err != nil {
		return nil, err
	}
	sort.Slice(list.Data, func(i, j int) bool { return list.Data[i].ID < list.Data[j].ID })
	return list.Data, nil
}

func (c *Client) get(ctx context.Context, url, project string, out any) error {
	tok, err := c.Tokens.Access(ctx)
	if err != nil {
		return err
	}
	resp, err := c.do(ctx, url, project, tok)
	if err != nil {
		return err
	}
	if resp.StatusCode == http.StatusUnauthorized {
		resp.Body.Close()
		if tok, err = c.Tokens.Refresh(ctx, tok); err != nil {
			return err
		}
		if resp, err = c.do(ctx, url, project, tok); err != nil {
			return err
		}
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		return &Error{Status: resp.StatusCode, Message: message(body, resp.Status)}
	}
	return json.Unmarshal(body, out)
}

func (c *Client) do(ctx context.Context, url, project, token string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Starlight-Client", "connect")
	if project != "" {
		req.Header.Set("X-Starlight-Project", project)
	}
	return c.HTTP.Do(req)
}

func message(body []byte, fallback string) string {
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal(body, &env) == nil {
		if env.Error.Message != "" {
			return env.Error.Message
		}
		if env.Message != "" {
			return env.Message
		}
	}
	return fallback
}

// IsUnauthorized reports whether err is Starlight refusing the credentials even after a refresh.
func IsUnauthorized(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Status == http.StatusUnauthorized
}
