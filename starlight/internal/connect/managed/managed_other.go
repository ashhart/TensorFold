//go:build !darwin && !windows

package managed

// SystemFile is the managed configuration path on Linux and other Unix systems.
var SystemFile = "/etc/tensorfold/connect.json"

func loadPlatform() (Policy, error) {
	return LoadJSONFiles(SystemFile)
}
