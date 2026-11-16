package services

import (
	"bytes"
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

//go:generate go tool mockgen -source=catalogs.go -destination=mocks/catalogs.go -package=mocks

// Source is the port the catalog is read through; Cached marks the adapter's disk copy of an unreachable URL.
type Source interface {
	Fetch(ctx context.Context) (Snapshot, error)
	Origin() string
}

// Snapshot is one read of the source.
type Snapshot struct {
	Body      []byte
	FetchedAt time.Time
	Cached    bool
}

// Freshness says where the numbers came from and whether they are a last good copy.
type Freshness struct {
	Origin    string    `json:"origin"`
	FetchedAt time.Time `json:"fetched_at"`
	// Stale: the source failed or broke a rule, so the last good copy is shown with Problem.
	Stale   bool   `json:"stale"`
	Problem string `json:"problem,omitempty"`
}

// CatalogService serves the last catalog that passed every rule; a bad read never replaces it.
type CatalogService struct {
	source Source
	log    *slog.Logger

	mu       sync.Mutex
	good     *Catalog
	goodBody []byte
	fresh    Freshness
}

// NewCatalogService returns a service reading source.
func NewCatalogService(source Source, log *slog.Logger) *CatalogService {
	return &CatalogService{source: source, log: log, fresh: Freshness{Origin: source.Origin()}}
}

// Board is the board for query from the current catalog.
func (service *CatalogService) Board(ctx context.Context, query validate.BoardQuery) (Board, error) {
	catalog, fresh, err := service.current(ctx)
	if err != nil {
		return Board{}, err
	}
	board, err := BuildBoard(catalog, query)
	board.Freshness = fresh
	return board, err
}

// Catalog is the current catalog as it was published.
func (service *CatalogService) Catalog(ctx context.Context) (Catalog, error) {
	catalog, _, err := service.current(ctx)
	return catalog, err
}

// Run is one run and its model by id.
func (service *CatalogService) Run(ctx context.Context, id string) (Row, error) {
	board, err := service.Board(ctx, validate.BoardQuery{})
	if err != nil {
		return Row{}, err
	}
	for _, row := range board.Rows {
		if row.Run.ID == id {
			return row, nil
		}
	}
	return Row{}, ErrRunNotFound
}

func (service *CatalogService) current(ctx context.Context) (Catalog, Freshness, error) {
	snapshot, fetchErr := service.source.Fetch(ctx)
	service.mu.Lock()
	defer service.mu.Unlock()
	switch {
	case fetchErr != nil:
		service.keepGood("source unreachable: "+fetchErr.Error(), snapshot.FetchedAt)
	case service.good != nil && bytes.Equal(snapshot.Body, service.goodBody):
		service.fresh = freshFrom(service.source.Origin(), snapshot)
	default:
		catalog, err := ParseCatalog(snapshot.Body)
		if err != nil {
			service.keepGood(err.Error(), snapshot.FetchedAt)
			break
		}
		service.good, service.goodBody = &catalog, snapshot.Body
		service.fresh = freshFrom(service.source.Origin(), snapshot)
	}
	if service.good == nil {
		return Catalog{}, service.fresh, ErrNoCatalogYet
	}
	return *service.good, service.fresh, nil
}

// freshFrom describes a snapshot that passed every rule.
func freshFrom(origin string, snapshot Snapshot) Freshness {
	fresh := Freshness{Origin: origin, FetchedAt: snapshot.FetchedAt, Stale: snapshot.Cached}
	if snapshot.Cached {
		fresh.Problem = "source unreachable: showing the cached copy"
	}
	return fresh
}

// keepGood records why the source was not used; the good copy stays.
func (service *CatalogService) keepGood(problem string, at time.Time) {
	if service.fresh.Problem != problem {
		service.log.Warn("models catalog: keeping the last good copy", "origin", service.source.Origin(), "problem", problem)
	}
	service.fresh.Stale = true
	service.fresh.Problem = problem
	if !at.IsZero() {
		service.fresh.FetchedAt = at
	}
}
