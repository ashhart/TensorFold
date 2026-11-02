package catalog_test

import (
	"context"
	"math"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestResolveThroughAliasesAndGrants(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	research, finance, qwen, small := ids.New(), ids.New(), ids.New(), ids.New()
	exec := func(sql string, args ...any) {
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	exec("INSERT INTO projects (id, slug, name) VALUES ($1,'research','R'), ($2,'finance','F')", research, finance)
	exec("INSERT INTO models (id, name, display_name) VALUES ($1,'qwen3.8-27b','Qwen'), ($2,'qwen3.5-2b','Small')", qwen, small)
	exec("INSERT INTO model_grants (id, model_id, project_id, max_output_tokens) VALUES ($1,$2,$3,2048)", ids.New(), qwen, research)
	exec("INSERT INTO model_grants (id, model_id) VALUES ($1,$2)", ids.New(), small)
	exec("INSERT INTO model_aliases (alias, model_id) VALUES ('claude-sonnet-4-5', $1)", small)
	exec("INSERT INTO model_aliases (alias, project_id, model_id) VALUES ('claude-sonnet-4-5', $1, $2)", research, qwen)
	c := &catalog.Catalog{Pool: pool}
	m, g, err := c.Resolve(ctx, research, "claude-sonnet-4-5")
	if err != nil || m.ID != qwen || g.MaxOutputTokens != 2048 {
		t.Fatalf("project alias: %+v %+v %v", m, g, err)
	}
	if m, _, err := c.Resolve(ctx, finance, "claude-sonnet-4-5"); err != nil || m.ID != small {
		t.Fatalf("global alias: %+v %v", m, err)
	}
	if _, _, err := c.Resolve(ctx, finance, "qwen3.8-27b"); err != catalog.ErrNotFound {
		t.Fatalf("ungranted model visible: %v", err)
	}
	if _, _, err := c.Resolve(ctx, finance, "nope"); err != catalog.ErrNotFound {
		t.Fatalf("unknown model: %v", err)
	}
	vis, err := c.Visible(ctx, finance)
	if err != nil || len(vis) != 1 || vis[0].Name != "qwen3.5-2b" {
		t.Fatalf("visible = %+v %v", vis, err)
	}
}

func TestCost(t *testing.T) {
	p := catalog.Prices{Input: 3_000_000, Output: 15_000_000, Cached: 300_000}
	if got := p.Cost(1_000_000, 100_000, 500_000); got != 1_500_000+150_000+1_500_000 {
		t.Fatalf("cost = %d", got)
	}
	absurd := catalog.Prices{Input: math.MaxInt64 / 2, Output: math.MaxInt64 / 2}
	if got := absurd.Cost(1_000_000, 1_000_000, 0); got != math.MaxInt64/1_000_000 {
		t.Fatalf("absurd prices cost %d instead of saturating", got)
	}
}
