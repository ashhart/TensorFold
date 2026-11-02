package service

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"time"
)

// KeepAwake holds a caffeinate assertion tied to this process, restarting it if it exits early.
func KeepAwake(ctx context.Context, log *slog.Logger) {
	for ctx.Err() == nil {
		cmd := exec.CommandContext(ctx, "/usr/bin/caffeinate", "-i", "-s", "-w", strconv.Itoa(os.Getpid()))
		err := cmd.Run()
		if ctx.Err() != nil {
			return
		}
		log.Warn("caffeinate exited; restarting it", "error", err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(10 * time.Second):
		}
	}
}
