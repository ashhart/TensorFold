package service

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Plist renders the launchd job; system scope is a LaunchDaemon running as a dedicated non-root user.
func Plist(o Options) (string, error) {
	if err := o.normalize(); err != nil {
		return "", err
	}
	var args strings.Builder
	for _, a := range append([]string{o.Binary}, o.Args...) {
		fmt.Fprintf(&args, "\t\t<string>%s</string>\n", xmlEscape(a))
	}
	logDir := "/Library/Logs/TensorFold"
	user := ""
	if o.Scope == "system" {
		if o.User == "" {
			return "", fmt.Errorf("a system LaunchDaemon needs --user; Starlight Node never runs as root")
		}
		user = fmt.Sprintf("\t<key>UserName</key>\n\t<string>%s</string>\n", xmlEscape(o.User))
	} else {
		logDir = filepath.Join(o.StateDir, "logs")
	}
	home := filepath.Join(o.StateDir, "home")
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>` + Label + `</string>
	<key>ProgramArguments</key>
	<array>
` + args.String() + `	</array>
` + user + `	<key>EnvironmentVariables</key>
	<dict>
		<key>HOME</key>
		<string>` + xmlEscape(home) + `</string>
		<key>PATH</key>
		<string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
	</dict>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>ProcessType</key>
	<string>Interactive</string>
	<key>ThrottleInterval</key>
	<integer>10</integer>
	<key>ExitTimeOut</key>
	<integer>180</integer>
	<key>SoftResourceLimits</key>
	<dict>
		<key>NumberOfFiles</key>
		<integer>65536</integer>
	</dict>
	<key>StandardOutPath</key>
	<string>` + xmlEscape(filepath.Join(logDir, "starlight-node.log")) + `</string>
	<key>StandardErrorPath</key>
	<string>` + xmlEscape(filepath.Join(logDir, "starlight-node.log")) + `</string>
</dict>
</plist>
`, nil
}
