// Package server holds the source adapters and wires the API and the UI on one mux.
package server

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/ashhart/TensorFold/web/models-preview/internal/api"
	"github.com/ashhart/TensorFold/web/models-preview/internal/config"
	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
	"github.com/ashhart/TensorFold/web/models-preview/ui"
)

// NewSource is the adapter cfg names: a URL with its HTTP cache, or a file.
func NewSource(cfg config.Config) services.Source {
	if cfg.Remote() {
		return &RemoteSource{
			URL:      cfg.Catalog,
			TTL:      cfg.TTL,
			CacheDir: cfg.CacheDir,
			Client:   &http.Client{Timeout: fetchTimeout},
			Now:      time.Now,
		}
	}
	return FileSource{Path: cfg.Catalog}
}

// Handler is every route: the UI's pages, the JSON API and the health check.
func Handler(cfg config.Config, log *slog.Logger) http.Handler {
	service := services.NewCatalogService(NewSource(cfg), log)
	catalogs := api.NewCatalogs(service)
	mux := http.NewServeMux()
	catalogs.Register(mux)
	mux.HandleFunc("GET "+config.AppRoutes.Health, func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusNoContent)
	})
	mux.Handle("/", ui.Handler(api.NewClient(catalogs), ui.Options{RepoURL: cfg.RepoURL, Log: log}))
	return mux
}
