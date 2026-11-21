package api_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"go.uber.org/mock/gomock"

	"github.com/ashhart/TensorFold/web/models-preview/internal/api"
	"github.com/ashhart/TensorFold/web/models-preview/internal/api/mocks"
	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

type envelope struct {
	Data    json.RawMessage   `json:"data"`
	Error   *string           `json:"error"`
	Details map[string]string `json:"details"`
}

func serve(t *testing.T, service api.CatalogService, path string) (int, envelope) {
	t.Helper()
	mux := http.NewServeMux()
	api.NewCatalogs(service).Register(mux)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, http.NoBody))
	var body envelope
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatalf("%s: not an envelope: %s", path, recorder.Body.String())
	}
	if recorder.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("%s: content type %q", path, recorder.Header().Get("Content-Type"))
	}
	return recorder.Code, body
}

func TestBoardEndpoint(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		path    string
		expect  func(*mocks.MockCatalogService)
		status  int
		details string
	}{
		"defaults and filters reach the service": {
			path: "/api/v1/board?hw=dgx-spark-2",
			expect: func(service *mocks.MockCatalogService) {
				want := validate.BoardQuery{Hardware: "dgx-spark-2", Sort: validate.SortRate, Dir: validate.Descending}
				service.EXPECT().Board(gomock.Any(), want).Return(services.Board{Updated: "2026-11-16"}, nil)
			},
			status: http.StatusOK,
		},
		"an unknown sort is a 400 with details": {
			path: "/api/v1/board?sort=speed", expect: func(*mocks.MockCatalogService) {}, status: http.StatusBadRequest, details: "sort",
		},
		"a malformed class is a 400": {
			path: "/api/v1/board?hw=DGX%20Spark", expect: func(*mocks.MockCatalogService) {}, status: http.StatusBadRequest, details: "hw",
		},
		"an unknown class is a 400": {
			path: "/api/v1/board?hw=tpu",
			expect: func(service *mocks.MockCatalogService) {
				service.EXPECT().Board(gomock.Any(), gomock.Any()).Return(services.Board{}, services.ErrUnknownFilter)
			},
			status: http.StatusBadRequest,
		},
		"no catalog yet is a 503": {
			path: "/api/v1/board",
			expect: func(service *mocks.MockCatalogService) {
				service.EXPECT().Board(gomock.Any(), gomock.Any()).Return(services.Board{}, services.ErrNoCatalogYet)
			},
			status: http.StatusServiceUnavailable,
		},
		"an unexpected error is a 500 without its text": {
			path: "/api/v1/board",
			expect: func(service *mocks.MockCatalogService) {
				service.EXPECT().Board(gomock.Any(), gomock.Any()).Return(services.Board{}, errors.New("disk on fire"))
			},
			status: http.StatusInternalServerError,
		},
		"the catalog as published": {
			path: "/api/v1/catalog",
			expect: func(service *mocks.MockCatalogService) {
				service.EXPECT().Catalog(gomock.Any()).Return(services.Catalog{SchemaVersion: 1}, nil)
			},
			status: http.StatusOK,
		},
		"one run": {
			path: "/api/v1/runs/beta-pair",
			expect: func(service *mocks.MockCatalogService) {
				service.EXPECT().Run(gomock.Any(), "beta-pair").Return(services.Row{}, nil)
			},
			status: http.StatusOK,
		},
		"a missing run is a 404": {
			path: "/api/v1/runs/nope",
			expect: func(service *mocks.MockCatalogService) {
				service.EXPECT().Run(gomock.Any(), "nope").Return(services.Row{}, services.ErrRunNotFound)
			},
			status: http.StatusNotFound,
		},
		"a malformed run id is a 400": {
			path: "/api/v1/runs/Bad_Id", expect: func(*mocks.MockCatalogService) {}, status: http.StatusBadRequest, details: "id",
		},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			service := mocks.NewMockCatalogService(gomock.NewController(t))
			test.expect(service)
			status, body := serve(t, service, test.path)
			if status != test.status {
				t.Fatalf("status %d, want %d", status, test.status)
			}
			if (status == http.StatusOK) != (body.Error == nil) || (status == http.StatusOK) == (string(body.Data) == "null") {
				t.Fatalf("envelope does not match the status: %+v", body)
			}
			if test.details != "" && body.Details[test.details] == "" {
				t.Fatalf("details %v lack %q", body.Details, test.details)
			}
			if status == http.StatusInternalServerError && *body.Error != "Internal server error" {
				t.Fatalf("a 500 shows %q", *body.Error)
			}
		})
	}
}

func TestClientCallsTheSameEndpoints(t *testing.T) {
	t.Parallel()
	service := mocks.NewMockCatalogService(gomock.NewController(t))
	client := api.NewClient(api.NewCatalogs(service))
	query := validate.BoardQuery{Mine: "apple-silicon", Sort: validate.SortTTFT, Dir: validate.Ascending}
	service.EXPECT().Board(gomock.Any(), query).Return(services.Board{Updated: "2026-11-16"}, nil)
	if board, err := client.Board(context.Background(), query); err != nil || board.Updated != "2026-11-16" {
		t.Fatalf("Board: %+v, %v", board, err)
	}
	service.EXPECT().Run(gomock.Any(), "a-run").Return(services.Row{Status: services.StatusSupported}, nil)
	if row, err := client.Run(context.Background(), "a-run"); err != nil || row.Status != services.StatusSupported {
		t.Fatalf("Run: %+v, %v", row, err)
	}
	var invalid *validate.Error
	if _, err := client.Board(context.Background(), validate.BoardQuery{Dir: "up"}); !errors.As(err, &invalid) {
		t.Fatalf("an invalid query reached the service: %v", err)
	}
}
