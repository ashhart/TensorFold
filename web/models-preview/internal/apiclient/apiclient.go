// Package apiclient is the API as the UI calls it; its types are aliases of what the API returns.
package apiclient

import (
	"context"

	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

// Client is every read the UI makes.
type Client interface {
	Board(ctx context.Context, query BoardQuery) (Board, error)
	Run(ctx context.Context, id string) (Row, error)
}

// Types the UI renders, as the API returns them.
type (
	Board         = services.Board
	Row           = services.Row
	Run           = services.Run
	Model         = services.Model
	Support       = services.Support
	Unmeasured    = services.Unmeasured
	HardwareClass = services.HardwareClass
	Freshness     = services.Freshness
	BoardQuery    = validate.BoardQuery
	ReadRun       = validate.ReadRun
	FieldErrors   = validate.Error
)

// Values the UI compares against.
const (
	ScopeAggregate     = services.ScopeAggregate
	ScopePerStream     = services.ScopePerStream
	WindowSteady       = services.WindowSteady
	StatusSupported    = services.StatusSupported
	StatusExperimental = services.StatusExperimental
	StatusPlanned      = services.StatusPlanned
	Ascending          = validate.Ascending
	Descending         = validate.Descending
	SortRate           = validate.SortRate
	SortModel          = validate.SortModel
	SortHardware       = validate.SortHardware
	SortTTFT           = validate.SortTTFT
	SortContext        = validate.SortContext
	SortConcurrency    = validate.SortConcurrency
	SortStatus         = validate.SortStatus
	SortDate           = validate.SortDate
)

// BoardQueryFrom reads a board query from a URL's query string.
var BoardQueryFrom = validate.BoardQueryFrom

// Error kinds the UI tells apart.
var (
	ErrUnknownFilter = services.ErrUnknownFilter
	ErrNoCatalogYet  = services.ErrNoCatalogYet
	ErrRunNotFound   = services.ErrRunNotFound
)
