//go:build darwin && cgo

package store

// #cgo LDFLAGS: -framework Security -framework CoreFoundation
import "C"
