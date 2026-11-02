package engine

import (
	"os"
	"sync"
)

// rotating is an append-only log that keeps the current file and two predecessors.
type rotating struct {
	mu   sync.Mutex
	path string
	max  int64
	f    *os.File
	size int64
}

func openRotating(path string, max int64) (*rotating, error) {
	if max <= 0 {
		max = 16 << 20
	}
	r := &rotating{path: path, max: max}
	return r, r.open()
}

func (r *rotating) open() error {
	f, err := os.OpenFile(r.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}
	r.f, r.size = f, fi.Size()
	return nil
}

func (r *rotating) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.f == nil {
		return len(p), nil
	}
	if r.size+int64(len(p)) > r.max {
		r.f.Close()
		_ = os.Rename(r.path+".1", r.path+".2")
		_ = os.Rename(r.path, r.path+".1")
		if err := r.open(); err != nil {
			r.f = nil
			return len(p), nil
		}
	}
	n, err := r.f.Write(p)
	r.size += int64(n)
	return n, err
}

func (r *rotating) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.f == nil {
		return nil
	}
	err := r.f.Close()
	r.f = nil
	return err
}
