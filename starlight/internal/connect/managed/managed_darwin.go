//go:build darwin

package managed

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"time"
)

// managedRoot is where macOS writes MDM-delivered preferences.
var managedRoot = "/Library/Managed Preferences"

func loadPlatform() (Policy, error) {
	paths := []string{filepath.Join(managedRoot, Domain+".plist")}
	if u, err := user.Current(); err == nil && u.Username != "" {
		paths = append(paths, filepath.Join(managedRoot, u.Username, Domain+".plist"))
	}
	return loadPlists(paths...)
}

// loadPlists converts each existing plist to JSON with plutil and merges them, later paths winning.
func loadPlists(paths ...string) (Policy, error) {
	var out Policy
	for _, path := range paths {
		if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		data, err := exec.CommandContext(ctx, "/usr/bin/plutil", "-convert", "json", "-o", "-", path).Output()
		cancel()
		if err != nil {
			return Policy{}, fmt.Errorf("managed preferences %s: %w", path, err)
		}
		p, err := ParseJSON(data, path)
		if err != nil {
			return Policy{}, err
		}
		out = Merge(out, p)
	}
	return out, nil
}
