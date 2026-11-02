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

func plistPath(scope string) (string, string, error) {
	if scope == "user" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", "", err
		}
		return filepath.Join(home, "Library", "LaunchAgents", Label+".plist"), "gui/" + strconv.Itoa(os.Getuid()), nil
	}
	return filepath.Join("/Library/LaunchDaemons", Label+".plist"), "system", nil
}

// Install writes the launchd job, prepares directories for the service user and loads it.
func Install(o Options) (string, error) {
	body, err := Plist(o)
	if err != nil {
		return "", err
	}
	if o.DryRun {
		return body, nil
	}
	path, domain, err := plistPath(o.Scope)
	if err != nil {
		return "", err
	}
	if o.Scope == "system" {
		if os.Geteuid() != 0 {
			return "", errors.New("installing a LaunchDaemon needs root: run with sudo")
		}
		u, err := user.Lookup(o.User)
		if err != nil {
			return "", fmt.Errorf("service user %q does not exist; create it first (see docs/nodes.md)", o.User)
		}
		uid, _ := strconv.Atoi(u.Uid)
		gid, _ := strconv.Atoi(u.Gid)
		for _, d := range []string{o.StateDir, filepath.Join(o.StateDir, "home"), "/Library/Logs/TensorFold"} {
			if err := os.MkdirAll(d, 0o700); err != nil {
				return "", err
			}
			if err := os.Chown(d, uid, gid); err != nil {
				return "", err
			}
		}
	} else if err := os.MkdirAll(filepath.Join(o.StateDir, "logs"), 0o700); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		return "", err
	}
	_ = exec.Command("/bin/launchctl", "bootout", domain+"/"+Label).Run()
	if out, err := exec.Command("/bin/launchctl", "bootstrap", domain, path).CombinedOutput(); err != nil {
		return "", fmt.Errorf("launchctl bootstrap: %v: %s", err, strings.TrimSpace(string(out)))
	}
	_ = exec.Command("/bin/launchctl", "enable", domain+"/"+Label).Run()
	return "installed " + path, nil
}

// Uninstall unloads the job and removes its plist; state and models stay.
func Uninstall(scope string) error {
	path, domain, err := plistPath(scope)
	if err != nil {
		return err
	}
	_ = exec.Command("/bin/launchctl", "bootout", domain+"/"+Label).Run()
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// Status reports launchd's view of the job.
func Status(scope string) (string, error) {
	_, domain, err := plistPath(scope)
	if err != nil {
		return "", err
	}
	out, err := exec.Command("/bin/launchctl", "print", domain+"/"+Label).CombinedOutput()
	if err != nil {
		return "not installed", nil
	}
	for _, line := range strings.Split(string(out), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "state = "); ok {
			return v, nil
		}
	}
	return "loaded", nil
}
