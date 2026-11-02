//go:build !windows

package engine_test

import "syscall"

func alive(pid int) bool { return pid > 0 && syscall.Kill(pid, 0) == nil }
