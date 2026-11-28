// Command preview serves the supported-models page from models.json, a file or an https URL (see README.md).
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ashhart/TensorFold/web/models-preview/internal/config"
	"github.com/ashhart/TensorFold/web/models-preview/internal/server"
)

const (
	// readHeaderTimeout bounds how long a client may take to send headers.
	readHeaderTimeout = 10 * time.Second
	// idleTimeout closes keep-alive connections nobody uses.
	idleTimeout = 2 * time.Minute
	// shutdownTimeout lets requests in flight finish on a signal.
	shutdownTimeout = 10 * time.Second
	// exitUsage is the exit status of a command-line misuse.
	exitUsage = 2
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	cfg, err := config.Load(os.Args[1:], os.Getenv, os.UserCacheDir, os.Stderr)
	if errors.Is(err, flag.ErrHelp) {
		return
	}
	if err != nil {
		log.Error("config", "err", err)
		os.Exit(exitUsage)
	}
	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           server.Handler(cfg, log),
		ReadHeaderTimeout: readHeaderTimeout,
		IdleTimeout:       idleTimeout,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()
	log.Info("models preview", "addr", cfg.Addr, "catalog", cfg.Catalog)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("serve", "err", err)
		os.Exit(1)
	}
}
