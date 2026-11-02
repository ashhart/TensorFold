// Package store keeps Connect's tokens in the operating system's credential store.
package store

import (
	"errors"
	"os"
)

// Service names Connect's entries in every credential store.
const Service = "TensorFold Connect"

// BackendEnv forces a backend; "file" keeps secrets in a 0600 file, for headless machines and tests.
const BackendEnv = "TENSORFOLD_CONNECT_STORE"

// ErrNotFound means the key holds nothing.
var ErrNotFound = errors.New("store: not found")

// Store holds small secrets by key.
type Store interface {
	Get(key string) ([]byte, error)
	Set(key string, value []byte) error
	Delete(key string) error
	Backend() string
}

// Open returns the platform credential store, falling back to a private file when none is usable.
func Open(dir string) Store {
	if os.Getenv(BackendEnv) == "file" {
		return NewFile(dir)
	}
	if s, err := openPlatform(); err == nil {
		return s
	}
	return NewFile(dir)
}
