//go:build darwin && cgo

package store

// #include "keychain_darwin.h"
import "C"

import (
	"fmt"
	"unsafe"
)

const errSecItemNotFound = -25300

// Keychain stores secrets as generic passwords in the user's login keychain.
type Keychain struct {
	service string
}

func openPlatform() (Store, error) { return &Keychain{service: Service}, nil }

// Backend names the store for status output.
func (k *Keychain) Backend() string { return "macOS Keychain" }

func (k *Keychain) names(key string) (*C.char, *C.char, func()) {
	svc, acc := C.CString(k.service), C.CString(key)
	return svc, acc, func() {
		C.free(unsafe.Pointer(svc))
		C.free(unsafe.Pointer(acc))
	}
}

// Get returns the value stored at key.
func (k *Keychain) Get(key string) ([]byte, error) {
	svc, acc, free := k.names(key)
	defer free()
	var out unsafe.Pointer
	var n C.int
	status := C.tf_keychain_get(svc, acc, &out, &n)
	if status == errSecItemNotFound {
		return nil, ErrNotFound
	}
	if status != 0 {
		return nil, fmt.Errorf("keychain read failed: OSStatus %d", int(status))
	}
	defer C.free(out)
	return C.GoBytes(out, n), nil
}

// Set stores value at key, replacing any earlier value.
func (k *Keychain) Set(key string, value []byte) error {
	svc, acc, free := k.names(key)
	defer free()
	ptr := C.CBytes(value)
	defer C.free(ptr)
	if status := C.tf_keychain_set(svc, acc, ptr, C.int(len(value))); status != 0 {
		return fmt.Errorf("keychain write failed: OSStatus %d", int(status))
	}
	return nil
}

// Delete removes key; deleting a missing key succeeds.
func (k *Keychain) Delete(key string) error {
	svc, acc, free := k.names(key)
	defer free()
	status := C.tf_keychain_delete(svc, acc)
	if status != 0 && status != errSecItemNotFound {
		return fmt.Errorf("keychain delete failed: OSStatus %d", int(status))
	}
	return nil
}
