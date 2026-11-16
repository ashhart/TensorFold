package services_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"go.uber.org/mock/gomock"

	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
	"github.com/ashhart/TensorFold/web/models-preview/internal/services/mocks"
	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

// step is one read of the source and what the board must show after it.
type step struct {
	body    []byte
	cached  bool
	err     error
	rows    int
	stale   bool
	wantErr error
}

func TestCatalogServiceServesOnlyGoodCatalogs(t *testing.T) {
	t.Parallel()
	good := encode(t, fixture())
	smaller := fixture()
	smaller.Runs = smaller.Runs[:1]
	broken := fixture()
	broken.Runs[1].RateWindow = ptr(services.WindowEndToEnd)
	unreachable := errors.New("dial tcp: no route")
	cases := map[string][]step{
		"nothing read yet":    {{err: unreachable, wantErr: services.ErrNoCatalogYet}},
		"a broken first copy": {{body: encode(t, broken), wantErr: services.ErrNoCatalogYet}},
		"good, then unreachable keeps it": {
			{body: good, rows: 3}, {err: unreachable, rows: 3, stale: true},
		},
		"good, then broken keeps it, then a new good one replaces it": {
			{body: good, rows: 3}, {body: encode(t, broken), rows: 3, stale: true}, {body: encode(t, smaller), rows: 1},
		},
		"the adapter's disk copy is marked stale": {
			{body: good, cached: true, rows: 3, stale: true}, {body: good, rows: 3},
		},
	}
	for name, steps := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			source := mocks.NewMockSource(gomock.NewController(t))
			source.EXPECT().Origin().Return("models.json").AnyTimes()
			service := services.NewCatalogService(source, quiet)
			for index, want := range steps {
				source.EXPECT().Fetch(gomock.Any()).
					Return(services.Snapshot{Body: want.body, FetchedAt: time.Unix(int64(index), 0), Cached: want.cached}, want.err)
				board, err := service.Board(context.Background(), validate.BoardQuery{})
				if !errors.Is(err, want.wantErr) {
					t.Fatalf("step %d: error %v, want %v", index, err, want.wantErr)
				}
				if err != nil {
					continue
				}
				if len(board.Rows) != want.rows || board.Freshness.Stale != want.stale {
					t.Fatalf("step %d: %d rows stale=%v, want %d stale=%v (%s)",
						index, len(board.Rows), board.Freshness.Stale, want.rows, want.stale, board.Freshness.Problem)
				}
				if want.stale && board.Freshness.Problem == "" {
					t.Fatalf("step %d: stale without a reason", index)
				}
			}
		})
	}
}

func TestCatalogServiceRunAndCatalog(t *testing.T) {
	t.Parallel()
	source := mocks.NewMockSource(gomock.NewController(t))
	source.EXPECT().Origin().Return("models.json").AnyTimes()
	source.EXPECT().Fetch(gomock.Any()).Return(services.Snapshot{Body: encode(t, fixture())}, nil).AnyTimes()
	service := services.NewCatalogService(source, quiet)
	row, err := service.Run(context.Background(), "beta-pair")
	if err != nil || row.Model.Name != "Beta" || row.Status != services.StatusExperimental {
		t.Fatalf("Run: %+v, %v", row, err)
	}
	if _, err := service.Run(context.Background(), "nope"); !errors.Is(err, services.ErrRunNotFound) {
		t.Fatalf("missing run: %v", err)
	}
	catalog, err := service.Catalog(context.Background())
	if err != nil || len(catalog.Models) != 2 {
		t.Fatalf("Catalog: %d models, %v", len(catalog.Models), err)
	}
}

func TestErrorKindsMatchTheirSentinels(t *testing.T) {
	t.Parallel()
	pairs := map[*services.Error]*services.Error{
		services.ErrRunNotFound:   services.ErrNotFound,
		services.ErrNoCatalogYet:  services.ErrUnavailable,
		services.ErrUnknownFilter: services.ErrValidation,
	}
	for err, kind := range pairs {
		if !errors.Is(err, kind) || errors.Is(err, services.ErrRunNotFound) && err != services.ErrRunNotFound {
			t.Errorf("%v does not match only its kind", err)
		}
		if err.Error() != err.Message {
			t.Errorf("%v: Error() is not the message", err)
		}
	}
}
