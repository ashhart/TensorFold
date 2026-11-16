// Package config reads flags and the environment once (a flag wins) and holds every route.
package config

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strings"
	"time"
)

// Defaults: the repository's own file from web/models-preview; LiveCatalog is its copy on main.
const (
	DefaultAddr     = ":3000"
	DefaultCatalog  = "../tensorfold_web/data/models.json"
	LiveCatalog     = "https://raw.githubusercontent.com/ashhart/TensorFold/main/web/tensorfold_web/data/models.json"
	DefaultTTL      = 5 * time.Minute
	DefaultRepoURL  = "https://github.com/ashhart/TensorFold#pick-a-model"
	cacheFolderName = "tensorfold-models-preview"
)

// Config is every setting the preview reads.
type Config struct {
	// Addr is the listen address; every interface by default, so the LAN and Tailscale reach it.
	Addr string
	// Catalog is a file path or an https URL of models.json.
	Catalog string
	// TTL is how long a fetched catalog is used before the URL is asked again.
	TTL time.Duration
	// CacheDir keeps the last fetched copy for a restart without network.
	CacheDir string
	// RepoURL is where "Full list of supported models" links.
	RepoURL string
}

// Remote reports whether the catalog is read over HTTP.
func (cfg Config) Remote() bool {
	return strings.HasPrefix(cfg.Catalog, "https://") || strings.HasPrefix(cfg.Catalog, "http://")
}

// Load parses args over the environment getenv reads; userCache is os.UserCacheDir.
func Load(args []string, getenv func(string) string, userCache func() (string, error), usage io.Writer) (Config, error) {
	envOr := func(name, fallback string) string {
		if value := strings.TrimSpace(getenv(name)); value != "" {
			return value
		}
		return fallback
	}
	ttl, err := time.ParseDuration(envOr("CATALOG_TTL", DefaultTTL.String()))
	if err != nil {
		return Config{}, fmt.Errorf("CATALOG_TTL: %w", err)
	}
	cacheDir := envOr("CATALOG_CACHE_DIR", "")
	if cacheDir == "" {
		base, err := userCache()
		if err != nil {
			return Config{}, fmt.Errorf("no cache folder (set CATALOG_CACHE_DIR): %w", err)
		}
		cacheDir = filepath.Join(base, cacheFolderName)
	}
	cfg := Config{}
	flags := flag.NewFlagSet("preview", flag.ContinueOnError)
	flags.SetOutput(usage)
	flags.StringVar(&cfg.Addr, "addr", envOr("PREVIEW_ADDR", DefaultAddr), "listen address (PREVIEW_ADDR)")
	flags.StringVar(&cfg.Catalog, "catalog", envOr("CATALOG", DefaultCatalog),
		"models.json: a file path or an https URL such as "+LiveCatalog+" (CATALOG)")
	flags.DurationVar(&cfg.TTL, "ttl", ttl, "how long a fetched catalog is reused before asking the URL again (CATALOG_TTL)")
	flags.StringVar(&cfg.CacheDir, "cache-dir", cacheDir, "where the last fetched copy is kept (CATALOG_CACHE_DIR)")
	flags.StringVar(&cfg.RepoURL, "repo-url", envOr("REPO_URL", DefaultRepoURL), "target of the full model list link (REPO_URL)")
	if err := flags.Parse(args); err != nil {
		return Config{}, err
	}
	return cfg, cfg.check()
}

func (cfg Config) check() error {
	var problems []error
	if cfg.TTL < 0 {
		problems = append(problems, errors.New("ttl must not be negative"))
	}
	if cfg.Remote() {
		if parsed, err := url.Parse(cfg.Catalog); err != nil || parsed.Host == "" {
			problems = append(problems, fmt.Errorf("catalog URL %q is not a URL", cfg.Catalog))
		}
	}
	if parsed, err := url.Parse(cfg.RepoURL); err != nil || parsed.Scheme != "https" {
		problems = append(problems, fmt.Errorf("repo-url %q must be an https URL", cfg.RepoURL))
	}
	return errors.Join(problems...)
}
