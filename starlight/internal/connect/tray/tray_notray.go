//go:build notray

package tray

import (
	"context"
	"errors"

	"github.com/ashhart/TensorFold/starlight/internal/connect/core"
)

// Run reports that this build was made without the tray.
func Run(ctx context.Context, c *core.Core) error {
	return errors.New("this build of TensorFold Connect has no tray; run `tensorfold-connect bridge` instead")
}
