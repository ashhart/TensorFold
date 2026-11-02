package service

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Unit renders a hardened systemd unit running the agent as an unprivileged user.
func Unit(o Options) (string, error) {
	if err := o.normalize(); err != nil {
		return "", err
	}
	if o.User == "" {
		return "", fmt.Errorf("the systemd unit needs --user; Starlight Node never runs as root")
	}
	words := make([]string, 0, len(o.Args)+1)
	for _, a := range append([]string{o.Binary}, o.Args...) {
		words = append(words, systemdQuote(a))
	}
	stateName := filepath.Base(o.StateDir)
	stateLine := "StateDirectory=" + stateName + "\nStateDirectoryMode=0700\n"
	if filepath.Dir(o.StateDir) != "/var/lib" {
		stateLine = "ReadWritePaths=" + systemdQuote(o.StateDir) + "\n"
	}
	return `[Unit]
Description=TensorFold Starlight Node
Documentation=https://tensorfold.dev
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=` + o.User + `
Group=` + o.User + `
ExecStart=` + strings.Join(words, " ") + `
Environment=HOME=` + systemdQuote(filepath.Join(o.StateDir, "home")) + `
` + stateLine + `Restart=on-failure
RestartSec=5
TimeoutStopSec=180
KillMode=mixed
LimitNOFILE=1048576
LimitMEMLOCK=infinity
UMask=0077
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
RestrictSUIDSGID=yes
RestrictRealtime=yes
RestrictNamespaces=yes
LockPersonality=yes
SystemCallArchitectures=native
CapabilityBoundingSet=
AmbientCapabilities=

[Install]
WantedBy=multi-user.target
`, nil
}
