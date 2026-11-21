package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/web/models-preview/internal/services"
)

// FileSource reads the catalog from a local file on every fetch (it is small), so an edit shows on the next page load.
type FileSource struct {
	Path string
}

var _ services.Source = FileSource{}

// Origin is the file's path.
func (source FileSource) Origin() string { return source.Path }

// Fetch reads the file; its modification time is the fetch time.
func (source FileSource) Fetch(context.Context) (services.Snapshot, error) {
	info, err := os.Stat(source.Path)
	if err != nil {
		return services.Snapshot{}, err
	}
	body, err := os.ReadFile(source.Path)
	if err != nil {
		return services.Snapshot{}, err
	}
	return services.Snapshot{Body: body, FetchedAt: info.ModTime()}, nil
}

const (
	// maxCatalogBytes caps a fetched catalog (today's is a few KiB).
	maxCatalogBytes = 4 << 20
	// fetchTimeout bounds one request to the catalog URL.
	fetchTimeout = 10 * time.Second
	// cacheFileMode keeps the cached copy readable by its owner only.
	cacheFileMode = 0o600
	// cacheDirMode is the cache folder's mode.
	cacheDirMode = 0o700
	// keyChars is how much of the URL's hash names its cache files.
	keyChars = 16
)

// RemoteSource fetches a URL with its ETag, reuses it for TTL, keeps it on disk, and serves it if the URL fails.
type RemoteSource struct {
	URL      string
	TTL      time.Duration
	CacheDir string
	Client   *http.Client
	Now      func() time.Time

	mu        sync.Mutex
	loaded    bool
	body      []byte
	etag      string
	fetchedAt time.Time
	checkedAt time.Time
	failed    bool
}

var _ services.Source = (*RemoteSource)(nil)

// cacheMeta is what is kept next to the cached body.
type cacheMeta struct {
	ETag      string    `json:"etag"`
	FetchedAt time.Time `json:"fetched_at"`
}

// Origin is the URL.
func (source *RemoteSource) Origin() string { return source.URL }

// Fetch returns the cached copy while it is fresh, else asks the URL.
func (source *RemoteSource) Fetch(ctx context.Context) (services.Snapshot, error) {
	source.mu.Lock()
	defer source.mu.Unlock()
	if !source.loaded {
		source.loaded = true
		source.readDisk()
	}
	now := source.Now()
	if source.body != nil && now.Sub(source.checkedAt) < source.TTL {
		return source.snapshot(), nil
	}
	source.checkedAt = now
	err := source.refresh(ctx, now)
	source.failed = err != nil
	if err != nil && source.body == nil {
		return services.Snapshot{}, err
	}
	return source.snapshot(), nil
}

func (source *RemoteSource) snapshot() services.Snapshot {
	return services.Snapshot{Body: source.body, FetchedAt: source.fetchedAt, Cached: source.failed}
}

func (source *RemoteSource) refresh(ctx context.Context, now time.Time) error {
	ctx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, source.URL, http.NoBody)
	if err != nil {
		return err
	}
	if source.etag != "" && source.body != nil {
		request.Header.Set("If-None-Match", source.etag)
	}
	response, err := source.Client.Do(request)
	if err != nil {
		return err
	}
	defer func() { _ = response.Body.Close() }()
	switch response.StatusCode {
	case http.StatusNotModified:
		source.fetchedAt = now
		return nil
	case http.StatusOK:
	default:
		return fmt.Errorf("GET %s: %s", source.URL, response.Status)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxCatalogBytes+1))
	if err != nil {
		return err
	}
	if len(body) > maxCatalogBytes {
		return errors.New("catalog is larger than 4 MiB")
	}
	source.body, source.etag, source.fetchedAt = body, response.Header.Get("ETag"), now
	source.writeDisk()
	return nil
}

func (source *RemoteSource) paths() (string, string) {
	sum := sha256.Sum256([]byte(source.URL))
	key := hex.EncodeToString(sum[:])[:keyChars]
	return filepath.Join(source.CacheDir, key+".json"), filepath.Join(source.CacheDir, key+".meta.json")
}

func (source *RemoteSource) readDisk() {
	bodyPath, metaPath := source.paths()
	body, err := os.ReadFile(bodyPath)
	if err != nil {
		return
	}
	var meta cacheMeta
	if raw, err := os.ReadFile(metaPath); err == nil {
		_ = json.Unmarshal(raw, &meta)
	}
	source.body, source.etag, source.fetchedAt = body, meta.ETag, meta.FetchedAt
}

// writeDisk keeps the copy for the next start; a failed write only costs that.
func (source *RemoteSource) writeDisk() {
	bodyPath, metaPath := source.paths()
	if err := os.MkdirAll(source.CacheDir, cacheDirMode); err != nil {
		return
	}
	meta, _ := json.Marshal(cacheMeta{ETag: source.etag, FetchedAt: source.fetchedAt})
	if writeAtomic(bodyPath, source.body) == nil {
		_ = writeAtomic(metaPath, meta)
	}
}

func writeAtomic(path string, data []byte) error {
	temp := path + ".part"
	if err := os.WriteFile(temp, data, cacheFileMode); err != nil {
		return err
	}
	return os.Rename(temp, path)
}
