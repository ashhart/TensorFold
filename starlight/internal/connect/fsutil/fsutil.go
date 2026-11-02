// Package fsutil writes Connect's private files atomically and serialises work across Connect processes.
package fsutil

import (
	"fmt"
	"os"
	"path/filepath"
)

// PrivateDir creates dir, and any parents, readable only by the current user.
func PrivateDir(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	return os.Chmod(dir, 0o700)
}

// WriteAtomic replaces path with data through a synced temporary file so readers never see a partial write.
func WriteAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := PrivateDir(dir); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	clean := func(err error) error {
		tmp.Close()
		os.Remove(name)
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		return clean(err)
	}
	if _, err := tmp.Write(data); err != nil {
		return clean(err)
	}
	if err := tmp.Sync(); err != nil {
		return clean(err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		os.Remove(name)
		return fmt.Errorf("replace %s: %w", filepath.Base(path), err)
	}
	return nil
}

// Lock takes an exclusive advisory lock on path, creating it, and returns the function that releases it.
func Lock(path string) (func(), error) {
	if err := PrivateDir(filepath.Dir(path)); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	if err := lockFile(f); err != nil {
		f.Close()
		return nil, fmt.Errorf("lock %s: %w", filepath.Base(path), err)
	}
	return func() {
		unlockFile(f)
		f.Close()
	}, nil
}
