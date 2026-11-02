// Package service installs Starlight Node as a system service and keeps the machine awake while it serves.
package service

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Name identifies the service on every platform.
const (
	Label       = "dev.tensorfold.starlight-node"
	UnitName    = "starlight-node.service"
	WindowsName = "TensorFoldStarlightNode"
)

// Options describe one installation.
type Options struct {
	Binary   string
	Args     []string
	User     string
	StateDir string
	Scope    string
	DryRun   bool
}

var userPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]{0,31}$`)

func (o *Options) normalize() error {
	if o.Binary == "" {
		exe, err := os.Executable()
		if err != nil {
			return err
		}
		o.Binary = exe
	}
	abs, err := filepath.Abs(o.Binary)
	if err != nil {
		return err
	}
	o.Binary = abs
	if o.Scope == "" {
		o.Scope = "system"
	}
	if o.Scope != "system" && o.Scope != "user" {
		return fmt.Errorf("scope must be system or user, got %q", o.Scope)
	}
	if o.User != "" && !userPattern.MatchString(o.User) {
		return fmt.Errorf("invalid service user %q", o.User)
	}
	for _, a := range append([]string{o.Binary, o.StateDir}, o.Args...) {
		if strings.ContainsAny(a, "\x00\n\r") {
			return errors.New("service arguments must not contain control characters")
		}
	}
	return nil
}

// RunArgs is the argument list the service starts the agent with.
func RunArgs(stateDir string, extra []string) []string {
	return append([]string{"run", "--state-dir", stateDir}, extra...)
}

func xmlEscape(s string) string {
	r := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;")
	return r.Replace(s)
}

// systemdQuote quotes one ExecStart word per systemd's rules.
func systemdQuote(s string) string {
	if s != "" && !strings.ContainsAny(s, " \t\"'\\$%;") {
		return s
	}
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "$", "$$", "%", "%%")
	return `"` + r.Replace(s) + `"`
}
