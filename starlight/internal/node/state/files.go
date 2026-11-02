package state

import (
	"os"
	"path/filepath"
	"runtime"
)

// WriteFileAtomic writes data to a temporary file beside path, syncs it and renames it into place.
func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(perm); err != nil && runtime.GOOS != "windows" {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return err
	}
	syncDir(dir)
	return nil
}

// DefaultRoot is the per-platform state directory used by the system service.
func DefaultRoot() string {
	switch runtime.GOOS {
	case "darwin":
		return "/Library/Application Support/TensorFold/Starlight Node"
	case "windows":
		base := os.Getenv("ProgramData")
		if base == "" {
			base = `C:\ProgramData`
		}
		return filepath.Join(base, "TensorFold", "StarlightNode")
	default:
		return "/var/lib/starlight-node"
	}
}
