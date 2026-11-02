//go:build windows

package store

import (
	"errors"
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	credTypeGeneric         = 1
	credPersistLocalMachine = 2
	credMaxBlob             = 5 * 512
)

var (
	advapi32   = windows.NewLazySystemDLL("advapi32.dll")
	credWriteW = advapi32.NewProc("CredWriteW")
	credReadW  = advapi32.NewProc("CredReadW")
	credDelete = advapi32.NewProc("CredDeleteW")
	credFree   = advapi32.NewProc("CredFree")
)

type credential struct {
	Flags              uint32
	Type               uint32
	TargetName         *uint16
	Comment            *uint16
	LastWritten        windows.Filetime
	CredentialBlobSize uint32
	CredentialBlob     *byte
	Persist            uint32
	AttributeCount     uint32
	Attributes         uintptr
	TargetAlias        *uint16
	UserName           *uint16
}

// WinCred stores secrets as generic credentials in Windows Credential Manager.
type WinCred struct {
	service string
}

func openPlatform() (Store, error) {
	if err := credReadW.Find(); err != nil {
		return nil, err
	}
	return &WinCred{service: Service}, nil
}

// Backend names the store for status output.
func (w *WinCred) Backend() string { return "Windows Credential Manager" }

func (w *WinCred) target(key string) (*uint16, error) {
	return windows.UTF16PtrFromString(w.service + "/" + key)
}

// Get returns the value stored at key.
func (w *WinCred) Get(key string) ([]byte, error) {
	target, err := w.target(key)
	if err != nil {
		return nil, err
	}
	var cred *credential
	r, _, callErr := credReadW.Call(uintptr(unsafe.Pointer(target)), credTypeGeneric, 0, uintptr(unsafe.Pointer(&cred)))
	if r == 0 {
		if errors.Is(callErr, windows.ERROR_NOT_FOUND) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("credential read failed: %w", callErr)
	}
	defer credFree.Call(uintptr(unsafe.Pointer(cred)))
	if cred.CredentialBlobSize == 0 || cred.CredentialBlob == nil {
		return []byte{}, nil
	}
	return append([]byte(nil), unsafe.Slice(cred.CredentialBlob, cred.CredentialBlobSize)...), nil
}

// Set stores value at key, replacing any earlier value.
func (w *WinCred) Set(key string, value []byte) error {
	if len(value) > credMaxBlob {
		return fmt.Errorf("credential of %d bytes exceeds the %d-byte limit", len(value), credMaxBlob)
	}
	target, err := w.target(key)
	if err != nil {
		return err
	}
	user, _ := windows.UTF16PtrFromString("tensorfold-connect")
	cred := credential{Type: credTypeGeneric, TargetName: target, Persist: credPersistLocalMachine, UserName: user,
		CredentialBlobSize: uint32(len(value))}
	if len(value) > 0 {
		cred.CredentialBlob = &value[0]
	}
	if r, _, callErr := credWriteW.Call(uintptr(unsafe.Pointer(&cred)), 0); r == 0 {
		return fmt.Errorf("credential write failed: %w", callErr)
	}
	return nil
}

// Delete removes key; deleting a missing key succeeds.
func (w *WinCred) Delete(key string) error {
	target, err := w.target(key)
	if err != nil {
		return err
	}
	if r, _, callErr := credDelete.Call(uintptr(unsafe.Pointer(target)), credTypeGeneric, 0); r == 0 && !errors.Is(callErr, windows.ERROR_NOT_FOUND) {
		return fmt.Errorf("credential delete failed: %w", callErr)
	}
	return nil
}
