// Package static embeds the page's browser assets: its stylesheet and the pinned Datastar bundle (no CDN).
package static

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"io/fs"
	"net/http"
	"strings"
)

// Datastar pin, from github.com/starfederation/datastar v1.0.4 bundles/datastar.js, as in Go Fast Stack.
const (
	DatastarFile   = "datastar-1.0.4.js"
	DatastarSHA256 = "727844adfc825ee651fb93c544a2a739986f9a21820a94524b35f0cac470cf91"
)

// CSSFile is the page's hand-written stylesheet.
const CSSFile = "models.css"

// prefix is where the assets are served (config.AppRoutes.Static).
const prefix = "/static/"

// versionChars is how much of an asset's hash its cache-busting query carries.
const versionChars = 16

//go:embed models.css datastar-1.0.4.js
var FS embed.FS

var versions = func() map[string]string {
	out := map[string]string{}
	_ = fs.WalkDir(FS, ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		file, err := FS.ReadFile(path)
		if err != nil {
			return err
		}
		sum := sha256.Sum256(file)
		out[path] = hex.EncodeToString(sum[:])[:versionChars]
		return nil
	})
	return out
}()

// URL returns the cache-busted URL of an embedded asset.
func URL(name string) string {
	return prefix + name + "?v=" + versions[name]
}

// Handler serves the embedded assets, immutable when the version matches, 404 for anything else.
func Handler() http.Handler {
	files := http.StripPrefix(prefix, http.FileServerFS(FS))
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		name := strings.TrimPrefix(request.URL.Path, prefix)
		if _, ok := versions[name]; !ok {
			http.NotFound(writer, request)
			return
		}
		if value := request.URL.Query().Get("v"); value != "" && value == versions[name] {
			writer.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		} else {
			writer.Header().Set("Cache-Control", "no-cache")
		}
		files.ServeHTTP(writer, request)
	})
}
