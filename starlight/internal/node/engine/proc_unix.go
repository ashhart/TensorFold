//go:build !windows

package engine

import (
	"os"
	"os/exec"
	"syscall"
)

func setProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	setDeathSignal(cmd.SysProcAttr)
}

// terminate signals the engine's whole process group.
func terminate(p *os.Process) {
	if err := syscall.Kill(-p.Pid, syscall.SIGTERM); err != nil {
		_ = p.Signal(syscall.SIGTERM)
	}
}

func kill(p *os.Process) {
	if err := syscall.Kill(-p.Pid, syscall.SIGKILL); err != nil {
		_ = p.Kill()
	}
}

// jobGroup is a no-op off Windows; process groups and the reaper cover orphan cleanup.
type jobGroup struct{}

func newJobGroup() (*jobGroup, error) { return &jobGroup{}, nil }

func (*jobGroup) add(*os.Process) error { return nil }
