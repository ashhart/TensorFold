//go:build !windows

package facts

import "syscall"

func diskFree(dir string) uint64 {
	var st syscall.Statfs_t
	if dir == "" || syscall.Statfs(dir, &st) != nil {
		return 0
	}
	return uint64(st.Bavail) * uint64(st.Bsize)
}
