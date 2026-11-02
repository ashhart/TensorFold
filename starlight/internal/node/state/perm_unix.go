//go:build !windows

package state

import (
	"fmt"
	"os"
	"syscall"
)

// EnsurePrivateDir creates path as 0700 and refuses a directory another account owns.
func EnsurePrivateDir(path string) error {
	if err := os.MkdirAll(path, 0o700); err != nil {
		return err
	}
	fi, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !fi.IsDir() {
		return fmt.Errorf("%s is not a directory", path)
	}
	if st, ok := fi.Sys().(*syscall.Stat_t); ok && int(st.Uid) != os.Geteuid() {
		return fmt.Errorf("%s is owned by uid %d, not this account (uid %d)", path, st.Uid, os.Geteuid())
	}
	if fi.Mode().Perm()&0o077 != 0 {
		return os.Chmod(path, 0o700)
	}
	return nil
}

func syncDir(dir string) {
	if f, err := os.Open(dir); err == nil {
		_ = f.Sync()
		f.Close()
	}
}
