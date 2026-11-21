// Package ui renders the page with templ and Datastar, reading data only through apiclient.Client.
package ui

import (
	"errors"
	"log/slog"
	"net/http"

	"github.com/starfederation/datastar-go/datastar"

	"github.com/ashhart/TensorFold/web/models-preview/internal/apiclient"
	"github.com/ashhart/TensorFold/web/models-preview/internal/config"
	"github.com/ashhart/TensorFold/web/models-preview/ui/static"
)

// Options are the page's settings.
type Options struct {
	// RepoURL is the "Full list of supported models" link.
	RepoURL string
	Log     *slog.Logger
	// HSTS sends Strict-Transport-Security (behind https only).
	HSTS bool
}

type pages struct {
	api     apiclient.Client
	repoURL string
	log     *slog.Logger
}

// Handler serves the page, its Datastar reads and its assets, with the security headers on every response.
func Handler(client apiclient.Client, options Options) http.Handler {
	ui := &pages{api: client, repoURL: options.RepoURL, log: options.Log}
	if ui.log == nil {
		ui.log = slog.Default()
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET "+config.AppRoutes.Home+"{$}", ui.home)
	mux.HandleFunc("GET "+config.AppRoutes.Board, ui.board)
	mux.Handle("GET "+config.AppRoutes.Static, static.Handler())
	return secure(strict(mux), options.HSTS)
}

// home renders the whole page for the query string.
func (ui *pages) home(writer http.ResponseWriter, request *http.Request) {
	query := apiclient.BoardQueryFrom(request.URL.Query())
	board, err := ui.api.Board(request.Context(), query)
	if status, ok := ui.failed(request, err); !ok {
		writer.WriteHeader(status)
		_ = Unavailable(ui.repoURL, err).Render(request.Context(), writer)
		return
	}
	_ = Page(board, ui.repoURL).Render(request.Context(), writer)
}

// board answers a Datastar read with the board fragment for the query.
func (ui *pages) board(writer http.ResponseWriter, request *http.Request) {
	query := apiclient.BoardQueryFrom(request.URL.Query())
	board, err := ui.api.Board(request.Context(), query)
	sse := datastar.NewSSE(writer, request)
	if _, ok := ui.failed(request, err); !ok {
		_ = sse.PatchElementTempl(Notice(errorMessage(err)))
		return
	}
	if err := sse.PatchElementTempl(BoardRoot(board, ui.repoURL)); err != nil {
		ui.log.WarnContext(request.Context(), "ui: patch failed", "err", err)
	}
}

// failed maps an API error to the page's status; ok is false when there is nothing to render.
func (ui *pages) failed(request *http.Request, err error) (int, bool) {
	var fields *apiclient.FieldErrors
	switch {
	case err == nil:
		return http.StatusOK, true
	case errors.As(err, &fields), errors.Is(err, apiclient.ErrUnknownFilter):
		return http.StatusBadRequest, false
	case errors.Is(err, apiclient.ErrNoCatalogYet):
		ui.log.WarnContext(request.Context(), "ui: no catalog", "err", err)
		return http.StatusServiceUnavailable, false
	default:
		ui.log.ErrorContext(request.Context(), "ui: request failed", "err", err)
		return http.StatusInternalServerError, false
	}
}

func errorMessage(err error) string {
	var fields *apiclient.FieldErrors
	switch {
	case errors.As(err, &fields), errors.Is(err, apiclient.ErrUnknownFilter):
		return "That filter or sort is not one this page knows. Showing nothing changed."
	case errors.Is(err, apiclient.ErrNoCatalogYet):
		return "The models catalog could not be read, and no earlier copy is cached."
	default:
		return "Something went wrong reading the models catalog."
	}
}
