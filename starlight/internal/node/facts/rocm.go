package facts

import (
	"context"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// amdGPUs lists the GPU agents rocminfo reports, less any the agent's own HIP device list withholds from Starlight.
func amdGPUs(ctx context.Context) ([]wire.GPU, error) {
	path, err := exec.LookPath("rocminfo")
	if err != nil {
		if path, err = exec.LookPath("/opt/rocm/bin/rocminfo"); err != nil {
			return nil, nil
		}
	}
	out, err := run(ctx, 20*time.Second, path)
	if err != nil {
		return nil, err
	}
	return hipVisible(parseRocminfo(out), hipList(os.LookupEnv)), nil
}

// rocmAgent is a GPU agent as Starlight hears of it, plus the Uuid text rocminfo printed, where HIP finds GPU- entries.
type rocmAgent struct {
	gpu  wire.GPU
	uuid string
}

// parseRocminfo reads rocminfo's GPU agents in its order, the order HIP numbers from 0; other agents are skipped.
func parseRocminfo(out string) []rocmAgent {
	var agents []rocmAgent
	printed := map[string]int{}
	for _, block := range strings.Split(out, "*******") {
		var a rocmAgent
		var gpu, apu, global bool
		var target, marketing, vendor string
		for _, line := range strings.Split(block, "\n") {
			key, value, ok := strings.Cut(line, ":")
			if !ok {
				continue
			}
			switch value = strings.TrimSpace(value); strings.TrimSpace(key) {
			case "Device Type":
				gpu = value == "GPU"
			case "Name":
				if target == "" {
					target = value
				}
			case "Marketing Name":
				marketing = value
			case "Vendor Name":
				vendor = value
			case "Uuid":
				a.uuid = value
			case "Memory Properties":
				apu = strings.Contains(value, "APU")
			case "Segment":
				global = strings.HasPrefix(value, "GLOBAL")
			case "Size":
				if global && a.gpu.MemoryBytes == 0 {
					a.gpu.MemoryBytes = kib(value)
				}
			}
		}
		if !gpu {
			continue
		}
		name := marketing
		if name == "" {
			name = target
		}
		if vendor != "" && !strings.HasPrefix(name, vendor) {
			name = vendor + " " + name
		}
		a.gpu.Index, a.gpu.Name = len(agents), name
		// an APU's GLOBAL pool is system memory, so Starlight is told 0 and treats the GPU as unified memory
		if apu || strings.HasSuffix(name, " Graphics") {
			a.gpu.MemoryBytes = 0
		}
		printed[a.uuid]++
		agents = append(agents, a)
	}
	// GPU-XX (ROCr's mark for a part with no unique ID) or a Uuid two agents print pins nothing; Starlight uses the index
	for i, a := range agents {
		id, ok := strings.CutPrefix(a.uuid, "GPU-")
		if ok && id != "" && strings.Trim(id, "0123456789abcdefABCDEF") == "" && printed[a.uuid] == 1 {
			agents[i].gpu.UUID = a.uuid
		}
	}
	return agents
}

// kib reads rocminfo's "50315264(0x2ffc000) KB" as bytes, or 0.
func kib(v string) uint64 {
	n, err := strconv.ParseUint(v[:len(v)-len(strings.TrimLeft(v, "0123456789"))], 10, 64)
	if err != nil || n >= 1<<43 || !strings.HasSuffix(v, "KB") {
		return 0
	}
	return n << 10
}

// hipList is the device list HIP reads: HIP_VISIBLE_DEVICES, else CUDA_VISIBLE_DEVICES; "" names every GPU.
func hipList(lookup func(string) (string, bool)) string {
	if list := clrFlag(lookup("HIP_VISIBLE_DEVICES")); list != "" {
		return list
	}
	return clrFlag(lookup("CUDA_VISIBLE_DEVICES"))
}

// clrFlag holds a variable as clr's flags do: unset is "", set but empty is one space, one pair of quotes is dropped.
func clrFlag(v string, set bool) string {
	q := strings.Trim(v, " ")
	switch {
	case !set:
		return ""
	case v == "":
		return " "
	case len(q) >= 2 && q[0] == '"' && q[len(q)-1] == '"':
		return q[1 : len(q)-1]
	}
	return v
}

// hipVisible keeps the GPUs a HIP device list names, as clr reads it: a plain index, or GPU- text in an agent's Uuid.
func hipVisible(agents []rocmAgent, list string) []wire.GPU {
	var out []wire.GPU
	if list == "" {
		for _, a := range agents {
			out = append(out, a.gpu)
		}
		return out
	}
	seen := map[int]bool{}
	for _, e := range strings.Split(list, ",") {
		i := -1
		if strings.Contains(e, "GPU-") {
			i = slices.IndexFunc(agents, func(a rocmAgent) bool { return strings.Contains(a.uuid, e) })
		} else if n, err := strconv.Atoi(e); err == nil && n >= 0 && n < len(agents) && strconv.Itoa(n) == e {
			i = n
		}
		if i < 0 {
			break
		}
		if !seen[i] {
			seen[i] = true
			out = append(out, agents[i].gpu)
		}
	}
	return out
}
