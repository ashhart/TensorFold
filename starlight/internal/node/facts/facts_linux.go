package facts

import (
	"bufio"
	"context"
	"os"
	"strconv"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

func memoryBytes(context.Context) uint64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	return parseMeminfo(bufio.NewScanner(f))
}

func parseMeminfo(sc *bufio.Scanner) uint64 {
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) >= 2 && fields[0] == "MemTotal:" {
			kb, err := strconv.ParseUint(fields[1], 10, 64)
			if err != nil {
				return 0
			}
			return kb << 10
		}
	}
	return 0
}

func osVersion(context.Context) string {
	name := ""
	if b, err := os.ReadFile("/etc/os-release"); err == nil {
		name = osRelease(string(b))
	}
	if k, err := os.ReadFile("/proc/sys/kernel/osrelease"); err == nil {
		return strings.TrimSpace(name + " kernel " + strings.TrimSpace(string(k)))
	}
	return name
}

func osRelease(body string) string {
	for _, line := range strings.Split(body, "\n") {
		if v, ok := strings.CutPrefix(line, "PRETTY_NAME="); ok {
			return strings.Trim(strings.TrimSpace(v), `"'`)
		}
	}
	return ""
}

// gpus asks nvidia-smi first and falls back to rocminfo's AMD GPUs; an error means a probe ran and failed.
func gpus(ctx context.Context, _ uint64) ([]wire.GPU, error) {
	if g, err := nvidiaGPUs(ctx); len(g) > 0 || err != nil {
		return g, err
	}
	return amdGPUs(ctx)
}
