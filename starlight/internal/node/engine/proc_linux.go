package engine

import "syscall"

// setDeathSignal ends the engine if the agent dies, so no engine outlives its supervisor.
func setDeathSignal(a *syscall.SysProcAttr) { a.Pdeathsig = syscall.SIGTERM }
