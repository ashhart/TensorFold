// Package buildinfo reports the version stamped into Starlight binaries.
package buildinfo

import "runtime/debug"

// Version and Commit are set with -ldflags "-X .../buildinfo.Version=..." by release builds.
var (
	Version = "0.0.0-dev"
	Commit  = ""
)

// String is "version (commit)" with the VCS revision filled from the build when not stamped.
func String() string {
	c := Commit
	if c == "" {
		if bi, ok := debug.ReadBuildInfo(); ok {
			for _, s := range bi.Settings {
				if s.Key == "vcs.revision" && len(s.Value) >= 12 {
					c = s.Value[:12]
				}
			}
		}
	}
	if c == "" {
		return Version
	}
	return Version + " (" + c + ")"
}
