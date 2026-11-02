// Command starlight is the TensorFold Starlight server and its administration tool.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/buildinfo"
	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

func main() {
	cli.Main(&cli.Command{Name: "starlight", Summary: "TensorFold Starlight: one governed endpoint for an organisation's private AI fleet.", Sub: []*cli.Command{
		serveCmd(), migrateCmd(), initCmd(), doctorCmd(), adminCmd(), keysCmd(), auditCmd(), nodesCmd(), applyCmd(),
		{Name: "version", Summary: "Print the version", Run: func(context.Context, *flag.FlagSet) error {
			fmt.Println("starlight", buildinfo.String())
			return nil
		}},
	}})
}

var configPath string

func withConfig(fs *flag.FlagSet) {
	fs.StringVar(&configPath, "config", os.Getenv("STARLIGHT_CONFIG"), "path to starlight.yaml (env STARLIGHT_CONFIG)")
}

func loadConfig() (config.Config, error) {
	return config.Load(configPath)
}

func logger(cfg config.Config) *slog.Logger {
	level := slog.LevelInfo
	switch strings.ToLower(cfg.Log.Level) {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	}
	opts := &slog.HandlerOptions{Level: level}
	if cfg.Log.Format == "text" {
		return slog.New(slog.NewTextHandler(os.Stderr, opts))
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, opts))
}

// openDB connects using the configuration without requiring an initialised installation.
func openDB(ctx context.Context) (config.Config, *pgxpool.Pool, error) {
	cfg, err := loadConfig()
	if err != nil {
		return cfg, nil, err
	}
	url, err := cfg.DatabaseURL()
	if err != nil {
		return cfg, nil, err
	}
	pool, err := db.Open(ctx, url, db.Options{MaxConns: 4, ApplicationName: "starlight-cli"})
	return cfg, pool, err
}
