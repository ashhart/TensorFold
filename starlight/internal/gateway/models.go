package gateway

import (
	"context"
	"net/http"
	"sort"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

type modelEntry struct {
	ID        string         `json:"id"`
	Object    string         `json:"object"`
	Created   int64          `json:"created"`
	OwnedBy   string         `json:"owned_by"`
	Starlight map[string]any `json:"starlight"`
}

type anthropicModel struct {
	Type        string `json:"type"`
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	CreatedAt   string `json:"created_at"`
}

// models lists what the caller may use in its project, aliases included, in the client's dialect.
func (g *Gateway) models(w http.ResponseWriter, r *http.Request) {
	caller, _, _, _, err := g.authenticate(r)
	if err != nil {
		g.refuse(w, httpx.OpenAI, err)
		return
	}
	ctx := r.Context()
	visible, err := g.Catalog.Visible(ctx, caller.P.ProjectID)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	aliases, err := g.aliases(ctx, caller.P.ProjectID)
	if err != nil {
		httpx.Write(w, httpx.OpenAI, err)
		return
	}
	byID := map[string]catalog.Model{}
	var names []string
	for _, m := range visible {
		if caller.AllowsModel(m.Name) {
			byID[m.ID] = m
			names = append(names, m.Name)
		}
	}
	entries := map[string]catalog.Model{}
	for _, m := range byID {
		entries[m.Name] = m
	}
	for alias, modelID := range aliases {
		if m, ok := byID[modelID]; ok {
			entries[alias] = m
			names = append(names, alias)
		}
	}
	names = dedupe(names)
	created := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if r.Header.Get("Anthropic-Version") != "" {
		out := make([]anthropicModel, 0, len(names))
		for _, n := range names {
			out = append(out, anthropicModel{Type: "model", ID: n, DisplayName: entries[n].DisplayName, CreatedAt: created.Format(time.RFC3339)})
		}
		first, last := "", ""
		if len(out) > 0 {
			first, last = out[0].ID, out[len(out)-1].ID
		}
		httpx.JSON(w, http.StatusOK, map[string]any{"data": out, "has_more": false, "first_id": first, "last_id": last})
		return
	}
	out := make([]modelEntry, 0, len(names))
	for _, n := range names {
		m := entries[n]
		starlight := map[string]any{"model": m.Name, "display_name": m.DisplayName, "kind": m.Kind, "route_policy": m.RoutePolicy}
		out = append(out, modelEntry{ID: n, Object: "model", Created: created.Unix(), OwnedBy: "starlight", Starlight: starlight})
	}
	httpx.JSON(w, http.StatusOK, map[string]any{"object": "list", "data": out})
}

func (g *Gateway) aliases(ctx context.Context, projectID string) (map[string]string, error) {
	rows, err := g.Pool.Query(ctx, `SELECT DISTINCT ON (alias) alias, model_id FROM model_aliases
		WHERE project_id = $1 OR project_id IS NULL ORDER BY alias, project_id NULLS LAST`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]string{}
	for rows.Next() {
		var a, m string
		if err := rows.Scan(&a, &m); err != nil {
			return nil, err
		}
		out[a] = m
	}
	return out, rows.Err()
}

func dedupe(in []string) []string {
	sort.Strings(in)
	out := in[:0]
	for i, s := range in {
		if i == 0 || s != in[i-1] {
			out = append(out, s)
		}
	}
	return out
}
