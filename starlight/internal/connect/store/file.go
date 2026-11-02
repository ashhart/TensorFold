package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fsutil"
)

// File keeps secrets in a JSON file readable only by the user, locked against other Connect processes.
type File struct {
	path string
	mu   sync.Mutex
}

// NewFile returns a file store in dir.
func NewFile(dir string) *File {
	return &File{path: filepath.Join(dir, "secrets.json")}
}

// Backend names the store for status output.
func (f *File) Backend() string { return "private file " + f.path }

type fileDoc struct {
	Version int               `json:"version"`
	Items   map[string][]byte `json:"items"`
}

func (f *File) read() (fileDoc, error) {
	doc := fileDoc{Version: 1, Items: map[string][]byte{}}
	data, err := os.ReadFile(f.path)
	if errors.Is(err, os.ErrNotExist) {
		return doc, nil
	}
	if err != nil {
		return doc, err
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return doc, fmt.Errorf("secrets file: %w", err)
	}
	if doc.Items == nil {
		doc.Items = map[string][]byte{}
	}
	return doc, nil
}

func (f *File) update(fn func(doc *fileDoc)) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	unlock, err := fsutil.Lock(f.path + ".lock")
	if err != nil {
		return err
	}
	defer unlock()
	doc, err := f.read()
	if err != nil {
		return err
	}
	fn(&doc)
	data, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	return fsutil.WriteAtomic(f.path, data, 0o600)
}

// Get returns the value stored at key.
func (f *File) Get(key string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	doc, err := f.read()
	if err != nil {
		return nil, err
	}
	v, ok := doc.Items[key]
	if !ok {
		return nil, ErrNotFound
	}
	return v, nil
}

// Set stores value at key.
func (f *File) Set(key string, value []byte) error {
	return f.update(func(doc *fileDoc) { doc.Items[key] = append([]byte(nil), value...) })
}

// Delete removes key; deleting a missing key succeeds.
func (f *File) Delete(key string) error {
	return f.update(func(doc *fileDoc) { delete(doc.Items, key) })
}
