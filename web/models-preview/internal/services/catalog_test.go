package services_test

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

func ptr[Value any](value Value) *Value { return &value }

// fixture: two models, three runs on two classes, one aggregate, one supported backend with no run.
func fixture() services.Catalog {
	recipe := ptr("https://example.org/recipe")
	return services.Catalog{
		SchemaVersion: 1,
		Updated:       "2026-11-16",
		HardwareClasses: []services.HardwareClass{
			{ID: "mac", Label: "Mac"}, {ID: "spark", Label: "Spark"}, {ID: "pair", Label: "Pair"},
		},
		Models: []services.Model{
			{ID: "alpha", Family: "a", Name: "Alpha", Checkpoint: "org/alpha", Support: []services.Support{
				{Backend: "metal", Status: services.StatusSupported, RecipeURL: recipe, Hardware: []string{"mac"}},
				{Backend: "cuda", Status: services.StatusSupported, RecipeURL: recipe, Hardware: []string{"spark"}},
			}},
			{ID: "beta", Family: "b", Name: "Beta", Checkpoint: "org/beta", Support: []services.Support{
				{Backend: "cuda", Status: services.StatusExperimental, RecipeURL: nil, Hardware: []string{"pair"}},
			}},
		},
		Runs: []services.Run{
			run("alpha-mac", "alpha", "metal", "mac", ptr(120.0), services.ScopePerStream, 1),
			run("beta-pair", "beta", "cuda", "pair", ptr(155.0), services.ScopeAggregate, 8),
			run("beta-pair-wait", "beta", "cuda", "pair", nil, "", 8),
		},
	}
}

func run(id, model, backend, class string, decode *float64, scope string, streams int) services.Run {
	out := services.Run{
		ID: id, ModelID: model, Backend: backend, Hardware: class + " box", HardwareClass: class,
		Concurrency: streams, DecodeTPS: decode, ReceiptURL: "https://example.org/receipt",
	}
	if decode != nil {
		out.RateScope = ptr(scope)
		out.RateSource = ptr(services.SourceClient)
		out.RateWindow = ptr(services.WindowDecode)
		if scope == services.ScopeAggregate {
			out.RateWindow = ptr(services.WindowSteady)
		}
	}
	return out
}

func encode(t *testing.T, catalog services.Catalog) []byte {
	t.Helper()
	body, err := json.Marshal(catalog)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestParseCatalogAcceptsTheRepositoryFile(t *testing.T) {
	t.Parallel()
	body, err := os.ReadFile("../../../tensorfold_web/data/models.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := services.ParseCatalog(body); err != nil {
		t.Fatal(err)
	}
}

func TestParseCatalogRefusesEachBrokenRule(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		breakIt func(*services.Catalog)
		want    string
	}{
		"makespan total":     {func(c *services.Catalog) { c.Runs[1].RateWindow = ptr(services.WindowEndToEnd) }, "steady-state"},
		"total of one":       {func(c *services.Catalog) { c.Runs[1].Concurrency = 1 }, "two or more"},
		"rate without scope": {func(c *services.Catalog) { c.Runs[0].RateScope = nil }, "no rate_scope"},
		"scope without rate": {func(c *services.Catalog) { c.Runs[2].RateScope = ptr(services.ScopePerStream) }, "does not have"},
		"unknown model":      {func(c *services.Catalog) { c.Runs[0].ModelID = "gamma" }, "unknown model"},
		"unknown class":      {func(c *services.Catalog) { c.Runs[0].HardwareClass = "tpu" }, "unknown hardware class"},
		"unlisted backend":   {func(c *services.Catalog) { c.Runs[0].Backend = "hip" }, "does not list"},
		"duplicate run":      {func(c *services.Catalog) { c.Runs[1].ID = "alpha-mac" }, "unique slug"},
		"duplicate model":    {func(c *services.Catalog) { c.Models[1].ID = "alpha" }, "appears twice"},
		"no support":         {func(c *services.Catalog) { c.Models[1].Support = nil }, "no backend support"},
		"bad status":         {func(c *services.Catalog) { c.Models[0].Support[0].Status = "beta" }, "known backend and status"},
		"zero rate":          {func(c *services.Catalog) { c.Runs[0].DecodeTPS = ptr(0.0) }, "not positive"},
		"plain http receipt": {func(c *services.Catalog) { c.Runs[0].ReceiptURL = "http://x" }, "https receipt"},
		"bad date":           {func(c *services.Catalog) { c.Runs[0].MeasuredAt = ptr("14 Nov") }, "YYYY-MM-DD"},
		"other version":      {func(c *services.Catalog) { c.SchemaVersion = 2 }, "schema_version"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			catalog := fixture()
			test.breakIt(&catalog)
			_, err := services.ParseCatalog(encode(t, catalog))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v, want an error naming %q", err, test.want)
			}
		})
	}
}

func TestParseCatalogRefusesUnknownKeys(t *testing.T) {
	t.Parallel()
	body := strings.Replace(string(encode(t, fixture())), `"decode_tps"`, `"decode_tsp"`, 1)
	if _, err := services.ParseCatalog([]byte(body)); err == nil {
		t.Fatal("a misspelt key was accepted")
	}
}

func ids(rows []services.Row) []string {
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.Run.ID)
	}
	return out
}

func TestBuildBoard(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		query      validate.BoardQuery
		rows       []string
		unmeasured []string
		fastest    string
	}{
		"fastest decode first, no rate last": {
			query: validate.BoardQuery{Sort: validate.SortRate, Dir: validate.Descending},
			rows:  []string{"beta-pair", "alpha-mac", "beta-pair-wait"}, unmeasured: []string{"alpha/cuda"}, fastest: "alpha-mac",
		},
		"ascending keeps no rate last": {
			query: validate.BoardQuery{Sort: validate.SortRate, Dir: validate.Ascending},
			rows:  []string{"alpha-mac", "beta-pair", "beta-pair-wait"}, unmeasured: []string{"alpha/cuda"}, fastest: "alpha-mac",
		},
		"filter keeps that class and its unmeasured support": {
			query: validate.BoardQuery{Hardware: "spark"},
			rows:  []string{}, unmeasured: []string{"alpha/cuda"}, fastest: "alpha-mac",
		},
		"an aggregate is never the fastest single stream": {
			query: validate.BoardQuery{Mine: "pair"},
			rows:  []string{"beta-pair", "alpha-mac", "beta-pair-wait"}, unmeasured: []string{"alpha/cuda"}, fastest: "",
		},
		"by model name": {
			query: validate.BoardQuery{Sort: validate.SortModel, Dir: validate.Ascending},
			rows:  []string{"alpha-mac", "beta-pair", "beta-pair-wait"}, unmeasured: []string{"alpha/cuda"}, fastest: "alpha-mac",
		},
		"by concurrency, most first": {
			query: validate.BoardQuery{Sort: validate.SortConcurrency, Dir: validate.Descending},
			rows:  []string{"beta-pair", "beta-pair-wait", "alpha-mac"}, unmeasured: []string{"alpha/cuda"}, fastest: "alpha-mac",
		},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			board, err := services.BuildBoard(fixture(), test.query)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Join(ids(board.Rows), ","); got != strings.Join(test.rows, ",") {
				t.Errorf("rows %s, want %s", got, strings.Join(test.rows, ","))
			}
			var missing []string
			for _, item := range board.Unmeasured {
				missing = append(missing, item.Model.ID+"/"+item.Support.Backend)
			}
			if got := strings.Join(missing, ","); got != strings.Join(test.unmeasured, ",") {
				t.Errorf("unmeasured %s, want %s", got, strings.Join(test.unmeasured, ","))
			}
			fastest := ""
			if board.Fastest != nil {
				fastest = board.Fastest.Run.ID
			}
			if fastest != test.fastest {
				t.Errorf("fastest %q, want %q", fastest, test.fastest)
			}
		})
	}
}

func TestBuildBoardStatusAndBars(t *testing.T) {
	t.Parallel()
	board, err := services.BuildBoard(fixture(), validate.BoardQuery{})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct {
		status string
		bar    int
	}{"beta-pair": {services.StatusExperimental, 100}, "alpha-mac": {services.StatusSupported, 77}, "beta-pair-wait": {services.StatusExperimental, 0}}
	for _, row := range board.Rows {
		if got := want[row.Run.ID]; row.Status != got.status || row.BarPercent != got.bar {
			t.Errorf("%s: status %q bar %d, want %q %d", row.Run.ID, row.Status, row.BarPercent, got.status, got.bar)
		}
	}
	if board.Query.Mine != "mac" {
		t.Errorf("the picker opens on %q, want the fastest single stream's class", board.Query.Mine)
	}
}

func TestBuildBoardRefusesAnUnknownClass(t *testing.T) {
	t.Parallel()
	for _, query := range []validate.BoardQuery{{Hardware: "tpu"}, {Mine: "tpu"}} {
		if _, err := services.BuildBoard(fixture(), query); !errors.Is(err, services.ErrUnknownFilter) {
			t.Errorf("%+v: got %v", query, err)
		}
	}
}
