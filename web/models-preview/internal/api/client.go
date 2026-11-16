package api

import (
	"context"

	"github.com/ashhart/TensorFold/web/models-preview/internal/apiclient"
)

// Client is apiclient.Client in-process: each method calls one Endpoint, with no network hop.
type Client struct {
	catalogs *Catalogs
}

var _ apiclient.Client = (*Client)(nil)

// NewClient returns the in-process client over the endpoints.
func NewClient(catalogs *Catalogs) *Client {
	return &Client{catalogs: catalogs}
}

// Board is GET /board.
func (client *Client) Board(ctx context.Context, query apiclient.BoardQuery) (apiclient.Board, error) {
	return client.catalogs.board.Call(ctx, query)
}

// Run is GET /runs/{id}.
func (client *Client) Run(ctx context.Context, id string) (apiclient.Row, error) {
	return client.catalogs.run.Call(ctx, apiclient.ReadRun{ID: id})
}
