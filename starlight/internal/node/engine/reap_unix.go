//go:build !windows

package engine

import (
	"bytes"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func commandLineContains(pid int, needle string) bool {
	if runtime.GOOS == "linux" {
		b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cmdline")
		return err == nil && bytes.Contains(bytes.ReplaceAll(b, []byte{0}, []byte{' '}), []byte(needle))
	}
	out, err := exec.Command("/bin/ps", "-o", "command=", "-p", strconv.Itoa(pid)).Output()
	return err == nil && strings.Contains(string(out), needle)
}

func stopPID(pid int, grace time.Duration) {
	if syscall.Kill(pid, syscall.SIGTERM) != nil {
		return
	}
	deadline := time.Now().Add(grace)
	for time.Now().Before(deadline) {
		if syscall.Kill(pid, 0) != nil {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	_ = syscall.Kill(pid, syscall.SIGKILL)
}
