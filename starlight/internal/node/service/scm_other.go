//go:build !windows

package service

import (
	"context"
	"errors"
)

// IsService is false outside Windows; launchd and systemd run the agent as an ordinary process.
func IsService() bool { return false }

// RunService exists only on Windows.
func RunService(func(context.Context) error) error {
	return errors.New("the Windows service runner is unavailable on this platform")
}
