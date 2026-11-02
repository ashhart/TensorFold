// Package appkeys issues the per-application keys local tools present to Connect's local endpoint.
package appkeys

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fsutil"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
)

// App is one local application allowed to use the local endpoint; only its key's hash is kept.
type App struct {
	Name     string    `json:"name"`
	Hash     string    `json:"hash"`
	PublicID string    `json:"public_id"`
	Created  time.Time `json:"created"`
}

// Errors returned by the registry.
var (
	ErrExists   = errors.New("an app with that name already exists")
	ErrNotFound = errors.New("no app with that name")
	ErrName     = errors.New("app names are 1-64 characters: lowercase letters, digits, '.', '_' or '-'")
)

var namePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

// ValidName reports whether name can label an app.
func ValidName(name string) bool { return namePattern.MatchString(name) }

// Registry is the app list in a 0600 file, reloaded when another process changes it.
type Registry struct {
	path  string
	mu    sync.Mutex
	apps  map[string]App
	stamp time.Time
	size  int64
}

// Open loads the registry at path, creating nothing until the first app is added.
func Open(path string) (*Registry, error) {
	r := &Registry{path: path, apps: map[string]App{}}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r, r.reload(true)
}

func (r *Registry) reload(force bool) error {
	fi, err := os.Stat(r.path)
	if errors.Is(err, os.ErrNotExist) {
		r.apps, r.stamp, r.size = map[string]App{}, time.Time{}, 0
		return nil
	}
	if err != nil {
		return err
	}
	if !force && fi.ModTime().Equal(r.stamp) && fi.Size() == r.size {
		return nil
	}
	data, err := os.ReadFile(r.path)
	if err != nil {
		return err
	}
	var list []App
	if err := json.Unmarshal(data, &list); err != nil {
		return fmt.Errorf("apps file: %w", err)
	}
	apps := make(map[string]App, len(list))
	for _, a := range list {
		apps[a.Name] = a
	}
	r.apps, r.stamp, r.size = apps, fi.ModTime(), fi.Size()
	return nil
}

func (r *Registry) change(fn func(apps map[string]App) error) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	unlock, err := fsutil.Lock(r.path + ".lock")
	if err != nil {
		return err
	}
	defer unlock()
	if err := r.reload(true); err != nil {
		return err
	}
	if err := fn(r.apps); err != nil {
		return err
	}
	list := make([]App, 0, len(r.apps))
	for _, a := range r.apps {
		list = append(list, a)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Name < list[j].Name })
	data, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	if err := fsutil.WriteAtomic(r.path, data, 0o600); err != nil {
		return err
	}
	return r.reload(true)
}

func mint(name string) (string, App) {
	tok, hash := secret.Generate(secret.AppKey)
	return tok, App{Name: name, Hash: hex.EncodeToString(hash), PublicID: secret.PublicID(tok), Created: time.Now().UTC()}
}

// Add creates an app and returns its key, which is shown once and never stored.
func (r *Registry) Add(name string) (string, error) {
	if !ValidName(name) {
		return "", ErrName
	}
	var tok string
	err := r.change(func(apps map[string]App) error {
		if _, ok := apps[name]; ok {
			return ErrExists
		}
		var a App
		tok, a = mint(name)
		apps[name] = a
		return nil
	})
	return tok, err
}

// Replace issues a new key for name, creating the app if needed; the old key stops working at once.
func (r *Registry) Replace(name string) (string, error) {
	if !ValidName(name) {
		return "", ErrName
	}
	var tok string
	err := r.change(func(apps map[string]App) error {
		var a App
		tok, a = mint(name)
		apps[name] = a
		return nil
	})
	return tok, err
}

// Remove deletes an app; its key stops working at once.
func (r *Registry) Remove(name string) error {
	return r.change(func(apps map[string]App) error {
		if _, ok := apps[name]; !ok {
			return ErrNotFound
		}
		delete(apps, name)
		return nil
	})
}

// List returns the apps sorted by name.
func (r *Registry) List() ([]App, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.reload(false); err != nil {
		return nil, err
	}
	out := make([]App, 0, len(r.apps))
	for _, a := range r.apps {
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Lookup finds the app a presented key belongs to.
func (r *Registry) Lookup(token string) (App, bool) {
	if k, ok := secret.Parse(token); !ok || k != secret.AppKey {
		return App{}, false
	}
	want := hex.EncodeToString(secret.Hash(token))
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.reload(false); err != nil {
		return App{}, false
	}
	for _, a := range r.apps {
		if secret.Equal([]byte(a.Hash), []byte(want)) {
			return a, true
		}
	}
	return App{}, false
}
