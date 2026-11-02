package starlightapi

import (
	"context"
	"net/http"
	"path/filepath"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fakestarlight"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
	"github.com/ashhart/TensorFold/starlight/internal/connect/tokens"
)

func TestMeAndModelsRefreshOnce(t *testing.T) {
	f := fakestarlight.New(t)
	dir := t.TempDir()
	m := tokens.New(store.NewFile(dir), filepath.Join(dir, "lock"), http.DefaultClient, f.Doc())
	tr := f.Issue()
	if err := m.Save(tr); err != nil {
		t.Fatal(err)
	}
	f.Expire(tr.AccessToken)
	c := &Client{HTTP: http.DefaultClient, Doc: f.Doc(), Tokens: m}
	me, err := c.Me(context.Background())
	if err != nil || me.Email != "ada@example.com" || len(me.Projects) != 1 {
		t.Fatalf("me = %+v, %v", me, err)
	}
	if f.RefreshCalls.Load() != 1 {
		t.Fatalf("%d refreshes after a 401", f.RefreshCalls.Load())
	}
	models, err := c.Models(context.Background(), "research")
	if err != nil || len(models) != 2 || models[0].ID != "nemotron-lightning" {
		t.Fatalf("models = %+v, %v", models, err)
	}
}
