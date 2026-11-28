package api

import (
	"context"
	"net/http"

	"github.com/ashhart/TensorFold/web/models-preview/internal/config"
	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

//go:generate go tool mockgen -source=catalog.go -destination=mocks/catalog.go -package=mocks

// CatalogService is what the catalog endpoints need from the service.
type CatalogService interface {
	Board(ctx context.Context, query validate.BoardQuery) (services.Board, error)
	Catalog(ctx context.Context) (services.Catalog, error)
	Run(ctx context.Context, id string) (services.Row, error)
}

var _ CatalogService = (*services.CatalogService)(nil)

// NoInput is the Input of an endpoint that takes none.
type NoInput struct{}

// Validate accepts the empty input.
func (*NoInput) Validate() error { return nil }

// Catalogs holds the catalog endpoints.
type Catalogs struct {
	board   Endpoint[validate.BoardQuery, services.Board]
	catalog Endpoint[NoInput, services.Catalog]
	run     Endpoint[validate.ReadRun, services.Row]
}

// NewCatalogs declares the endpoints over service.
func NewCatalogs(service CatalogService) *Catalogs {
	return &Catalogs{
		board: Endpoint[validate.BoardQuery, services.Board]{
			Method: http.MethodGet, Path: config.APIRoutes.Board,
			Decode: func(request *http.Request) validate.BoardQuery { return validate.BoardQueryFrom(request.URL.Query()) },
			Handle: service.Board,
		},
		catalog: Endpoint[NoInput, services.Catalog]{
			Method: http.MethodGet, Path: config.APIRoutes.Catalog,
			Decode: func(*http.Request) NoInput { return NoInput{} },
			Handle: func(ctx context.Context, _ NoInput) (services.Catalog, error) { return service.Catalog(ctx) },
		},
		run: Endpoint[validate.ReadRun, services.Row]{
			Method: http.MethodGet, Path: config.APIRoutes.Run,
			Decode: func(request *http.Request) validate.ReadRun { return validate.ReadRun{ID: request.PathValue("id")} },
			Handle: func(ctx context.Context, input validate.ReadRun) (services.Row, error) {
				return service.Run(ctx, input.ID)
			},
		},
	}
}

// Register mounts every endpoint on mux.
func (catalogs *Catalogs) Register(mux *http.ServeMux) {
	mux.Handle(catalogs.board.Pattern(), catalogs.board.Serve())
	mux.Handle(catalogs.catalog.Pattern(), catalogs.catalog.Serve())
	mux.Handle(catalogs.run.Pattern(), catalogs.run.Serve())
}
