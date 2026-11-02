// Package facts reports what a node is: platform, memory, accelerators, disk and engine version.
package facts

import (
	"bufio"
	"bytes"
	"context"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/buildinfo"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// Collect gathers every fact once; an agent that reports repeatedly uses a Collector.
func Collect(ctx context.Context, engine, dir string) wire.Facts {
	return new(Collector).Collect(ctx, engine, dir)
}

// Collector gathers facts again and again, keeping the last good GPU list through a failed GPU probe.
type Collector struct {
	mu    sync.Mutex
	gpus  []wire.GPU
	probe func(context.Context, uint64) ([]wire.GPU, error)
}

// Collect gathers every fact: a failed GPU probe keeps the last good list, other failed probes leave fields empty.
func (c *Collector) Collect(ctx context.Context, engine, dir string) wire.Facts {
	f := wire.Facts{Platform: runtime.GOOS, Arch: runtime.GOARCH, AgentVersion: buildinfo.Version}
	f.Hostname, _ = os.Hostname()
	f.MemoryBytes = memoryBytes(ctx)
	f.OSVersion = osVersion(ctx)
	f.GPUs = c.gpuList(ctx, f.MemoryBytes)
	f.Backend = Backend(f)
	f.EngineVersion = EngineVersion(ctx, engine)
	f.DiskFreeBytes = diskFree(dir)
	return f
}

// gpuList probes the GPUs, so that one nvidia-smi that times out never makes a GPU server look like unified memory.
func (c *Collector) gpuList(ctx context.Context, memory uint64) []wire.GPU {
	probe := c.probe
	if probe == nil {
		probe = gpus
	}
	g, err := probe(ctx, memory)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		return c.gpus
	}
	c.gpus = g
	return g
}

// Backend names the accelerator stack the engine will use on this machine.
func Backend(f wire.Facts) string {
	switch {
	case f.Platform == "darwin" && f.Arch == "arm64":
		return "metal"
	case len(f.GPUs) > 0 && strings.Contains(strings.ToUpper(f.GPUs[0].Name), "NVIDIA"):
		return "cuda"
	case f.Platform == "linux" && len(f.GPUs) > 0 && strings.HasPrefix(strings.ToUpper(f.GPUs[0].Name), "AMD "):
		return "hip"
	default:
		return ""
	}
}

// EngineVersion runs `engine --version` and returns its first line.
func EngineVersion(ctx context.Context, engine string) string {
	if engine == "" {
		return ""
	}
	out, err := run(ctx, 15*time.Second, engine, "--version")
	if err != nil {
		return ""
	}
	line, _, _ := strings.Cut(out, "\n")
	return strings.TrimSpace(line)
}

func run(ctx context.Context, timeout time.Duration, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	var out bytes.Buffer
	cmd.Stdout = &limited{w: &out, n: 1 << 20}
	if err := cmd.Run(); err != nil {
		return "", err
	}
	return strings.TrimSpace(out.String()), nil
}

type limited struct {
	w *bytes.Buffer
	n int
}

func (l *limited) Write(p []byte) (int, error) {
	if room := l.n - l.w.Len(); room > 0 {
		if len(p) > room {
			l.w.Write(p[:room])
		} else {
			l.w.Write(p)
		}
	}
	return len(p), nil
}

// nvidiaGPUs lists the GPUs nvidia-smi reports, less any the agent's own CUDA_VISIBLE_DEVICES withholds from Starlight.
func nvidiaGPUs(ctx context.Context) ([]wire.GPU, error) {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil, nil
	}
	out, err := run(ctx, 20*time.Second, path, "--query-gpu=index,uuid,memory.total,name", "--format=csv,noheader,nounits")
	if err != nil {
		return nil, err
	}
	list, set := os.LookupEnv("CUDA_VISIBLE_DEVICES")
	return visible(parseNvidiaCSV(out), list, set), nil
}

// parseNvidiaCSV reads "index, uuid, memory MiB, name" rows; unified-memory parts report memory as [N/A].
func parseNvidiaCSV(out string) []wire.GPU {
	var gpus []wire.GPU
	sc := bufio.NewScanner(strings.NewReader(out))
	for sc.Scan() {
		f := strings.SplitN(sc.Text(), ",", 4)
		if len(f) != 4 || strings.TrimSpace(f[3]) == "" {
			continue
		}
		index, err := strconv.Atoi(strings.TrimSpace(f[0]))
		if err != nil {
			continue
		}
		g := wire.GPU{Index: index, Name: strings.TrimSpace(f[3])}
		if id := strings.TrimSpace(f[1]); strings.HasPrefix(id, "GPU-") {
			g.UUID = id
		}
		if mib, err := strconv.ParseUint(strings.TrimSpace(f[2]), 10, 64); err == nil {
			g.MemoryBytes = mib << 20
		}
		gpus = append(gpus, g)
	}
	return gpus
}

// visible keeps the GPUs a CUDA_VISIBLE_DEVICES list names, by index or UUID prefix, as CUDA itself reads it.
func visible(gpus []wire.GPU, list string, set bool) []wire.GPU {
	if !set {
		return gpus
	}
	var out []wire.GPU
	seen := map[int]bool{}
	for _, e := range strings.Split(list, ",") {
		e = strings.TrimSpace(e)
		i := -1
		if n, err := strconv.Atoi(e); err == nil {
			for j, g := range gpus {
				if g.Index == n {
					i = j
					break
				}
			}
		} else if strings.HasPrefix(e, "GPU-") {
			for j, g := range gpus {
				if g.UUID != "" && strings.HasPrefix(g.UUID, e) {
					i = j
					break
				}
			}
		}
		if i < 0 {
			break
		}
		if !seen[i] {
			seen[i] = true
			out = append(out, gpus[i])
		}
	}
	return out
}
