//go:build !darwin && !windows

package service

import (
	"context"
	"log/slog"
)

// KeepAwake is a no-op on Linux; servers should disable suspend in their OS configuration.
func KeepAwake(ctx context.Context, log *slog.Logger) {
	log.Info("sleep prevention is not needed on this platform; disable suspend in the OS if the host can sleep")
	<-ctx.Done()
}
