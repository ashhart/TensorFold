//go:build (darwin && !cgo) || (!darwin && !windows && !linux)

package store

import "errors"

func openPlatform() (Store, error) {
	return nil, errors.New("no credential store in this build")
}
