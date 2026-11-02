package service

import (
	"context"
	"log/slog"
	"runtime"

	"golang.org/x/sys/windows"
)

const (
	esContinuous     = 0x80000000
	esSystemRequired = 0x00000001
)

var setThreadExecutionState = windows.NewLazySystemDLL("kernel32.dll").NewProc("SetThreadExecutionState")

// KeepAwake asks Windows not to sleep while ctx lives; the request belongs to one locked OS thread.
func KeepAwake(ctx context.Context, log *slog.Logger) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if r, _, err := setThreadExecutionState.Call(esContinuous | esSystemRequired); r == 0 {
		log.Warn("could not prevent sleep", "error", err)
		return
	}
	<-ctx.Done()
	_, _, _ = setThreadExecutionState.Call(esContinuous)
}
