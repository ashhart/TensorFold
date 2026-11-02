package facts

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

func memoryBytes(ctx context.Context) uint64 {
	out, err := run(ctx, 5*time.Second, "/usr/sbin/sysctl", "-n", "hw.memsize")
	if err != nil {
		return 0
	}
	n, _ := strconv.ParseUint(strings.TrimSpace(out), 10, 64)
	return n
}

func osVersion(ctx context.Context) string {
	v, err := run(ctx, 5*time.Second, "/usr/bin/sw_vers", "-productVersion")
	if err != nil {
		return ""
	}
	if b, err := run(ctx, 5*time.Second, "/usr/bin/sw_vers", "-buildVersion"); err == nil && b != "" {
		return "macOS " + v + " (" + b + ")"
	}
	return "macOS " + v
}

// gpus reports the Apple chip as one accelerator sharing unified memory.
func gpus(ctx context.Context, memory uint64) ([]wire.GPU, error) {
	chip := chipName(ctx)
	if chip == "" {
		return nil, errors.New("system_profiler named no chip")
	}
	return []wire.GPU{{Name: chip, MemoryBytes: memory}}, nil
}

func chipName(ctx context.Context) string {
	if out, err := run(ctx, 20*time.Second, "/usr/sbin/system_profiler", "SPHardwareDataType", "-json"); err == nil {
		var doc struct {
			Items []struct {
				Chip string `json:"chip_type"`
				CPU  string `json:"cpu_type"`
			} `json:"SPHardwareDataType"`
		}
		if json.Unmarshal([]byte(out), &doc) == nil && len(doc.Items) > 0 {
			if doc.Items[0].Chip != "" {
				return doc.Items[0].Chip
			}
			if doc.Items[0].CPU != "" {
				return doc.Items[0].CPU
			}
		}
	}
	out, err := run(ctx, 5*time.Second, "/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
	if err != nil {
		return ""
	}
	return out
}
