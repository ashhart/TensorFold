package service

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
)

const unitPath = "/etc/systemd/system/" + UnitName

// Install writes the systemd unit, prepares the state directory and starts the service.
func Install(o Options) (string, error) {
	if o.Scope == "user" {
		return "", errors.New("on Linux Starlight Node installs as a system service only")
	}
	body, err := Unit(o)
	if err != nil {
		return "", err
	}
	if o.DryRun {
		return body, nil
	}
	if os.Geteuid() != 0 {
		return "", errors.New("installing a systemd unit needs root: run with sudo")
	}
	u, err := user.Lookup(o.User)
	if err != nil {
		return "", fmt.Errorf("service user %q does not exist; create it first (see docs/nodes.md)", o.User)
	}
	uid, _ := strconv.Atoi(u.Uid)
	gid, _ := strconv.Atoi(u.Gid)
	for _, d := range []string{o.StateDir, filepath.Join(o.StateDir, "home")} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			return "", err
		}
		if err := os.Chown(d, uid, gid); err != nil {
			return "", err
		}
	}
	if err := os.WriteFile(unitPath, []byte(body), 0o644); err != nil {
		return "", err
	}
	for _, args := range [][]string{{"daemon-reload"}, {"enable", "--now", UnitName}} {
		if out, err := exec.Command("systemctl", args...).CombinedOutput(); err != nil {
			return "", fmt.Errorf("systemctl %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
		}
	}
	return "installed " + unitPath, nil
}

// Uninstall stops and removes the unit; state and models stay.
func Uninstall(string) error {
	_ = exec.Command("systemctl", "disable", "--now", UnitName).Run()
	if err := os.Remove(unitPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return exec.Command("systemctl", "daemon-reload").Run()
}

// Status reports systemd's active state.
func Status(string) (string, error) {
	out, _ := exec.Command("systemctl", "is-active", UnitName).Output()
	s := strings.TrimSpace(string(out))
	if s == "" {
		s = "not installed"
	}
	return s, nil
}
