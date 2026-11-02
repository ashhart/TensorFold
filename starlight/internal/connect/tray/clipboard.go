package tray

import (
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// CopyText puts s on the clipboard through the platform tool's standard input, so secrets never reach argv.
func CopyText(s string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("pbcopy")
	case "windows":
		cmd = exec.Command("clip")
	default:
		if os.Getenv("WAYLAND_DISPLAY") != "" {
			if p, err := exec.LookPath("wl-copy"); err == nil {
				cmd = exec.Command(p)
			}
		}
		if cmd == nil {
			if p, err := exec.LookPath("xclip"); err == nil {
				cmd = exec.Command(p, "-selection", "clipboard")
			}
		}
		if cmd == nil {
			return errors.New("install wl-copy or xclip to copy setup text")
		}
	}
	cmd.Stdin = strings.NewReader(s)
	return cmd.Run()
}
