package engine

import (
	"context"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// MinEngine is the oldest TensorFold release Starlight runs: the one it was qualified on, which has scoped engine keys.
var MinEngine = [3]int{1, 2, 0}

var versionPattern = regexp.MustCompile(`(\d+)\.(\d+)\.(\d+)`)

// parseVersion reads the first x.y.z in an engine's --version line.
func parseVersion(line string) ([3]int, bool) {
	m := versionPattern.FindStringSubmatch(line)
	if m == nil {
		return [3]int{}, false
	}
	var v [3]int
	for i := range v {
		v[i], _ = strconv.Atoi(m[i+1])
	}
	return v, true
}

func older(a, b [3]int) bool {
	for i := range a {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}

// checkEngine refuses an engine older than MinEngine; releases before 1.1.8 would also give every key full access.
func checkEngine(ctx context.Context, bin string) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, bin, "--version").Output()
	if err != nil {
		return fmt.Errorf("engine %s did not report a version: %w", bin, err)
	}
	line, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
	if v, ok := parseVersion(line); ok && older(v, MinEngine) {
		return fmt.Errorf("engine %q is older than %d.%d.%d, the first TensorFold release Starlight runs; upgrade TensorFold on this node",
			line, MinEngine[0], MinEngine[1], MinEngine[2])
	}
	return nil
}
