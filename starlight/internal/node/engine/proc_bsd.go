//go:build !windows && !linux

package engine

import "syscall"

// setDeathSignal has no equivalent here; the reaper kills orphans from pid files at the next start.
func setDeathSignal(*syscall.SysProcAttr) {}
