package facts

import (
	"context"
	"errors"
	"runtime"
	"slices"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

func TestParseNvidiaCSV(t *testing.T) {
	got := parseNvidiaCSV("0, GPU-1b2c3d4e-0000-4000-8000-000000000001, 288000, NVIDIA B300\n1, [N/A], [N/A], NVIDIA GB10, rev 2\n\n")
	if len(got) != 2 || got[0].Name != "NVIDIA B300" || got[0].MemoryBytes != 288000<<20 || got[0].UUID == "" ||
		got[1].Index != 1 || got[1].MemoryBytes != 0 || got[1].UUID != "" || got[1].Name != "NVIDIA GB10, rev 2" {
		t.Fatalf("parsed %+v", got)
	}
}

func TestVisibleFollowsCUDARules(t *testing.T) {
	gpus := []wire.GPU{{Index: 0, Name: "a", UUID: "GPU-aaaa1111"}, {Index: 1, Name: "b", UUID: "GPU-bbbb2222"}, {Index: 2, Name: "c", UUID: "GPU-cccc3333"}}
	names := func(gs []wire.GPU) (s string) {
		for _, g := range gs {
			s += g.Name
		}
		return s
	}
	for _, c := range []struct {
		list string
		set  bool
		want string
	}{
		{"", false, "abc"}, {"", true, ""}, {"2,0", true, "ca"}, {"GPU-bbbb,1,9,0", true, "b"}, {"1,1,0", true, "ba"}, {"MIG-x", true, ""},
	} {
		if got := names(visible(gpus, c.list, c.set)); got != c.want {
			t.Errorf("visible(%q, %v) = %q, want %q", c.list, c.set, got, c.want)
		}
	}
}

func TestBackend(t *testing.T) {
	cases := []struct {
		f    wire.Facts
		want string
	}{
		{wire.Facts{Platform: "darwin", Arch: "arm64"}, "metal"},
		{wire.Facts{Platform: "linux", GPUs: []wire.GPU{{Name: "NVIDIA RTX PRO 6000"}}}, "cuda"},
		{wire.Facts{Platform: "linux", GPUs: []wire.GPU{{Name: "AMD Instinct MI300X"}}}, "hip"},
		{wire.Facts{Platform: "linux", GPUs: []wire.GPU{{Name: "AMD Radeon 8060S Graphics"}}}, "hip"},
		{wire.Facts{Platform: "darwin", Arch: "amd64", GPUs: []wire.GPU{{Name: "AMD Ryzen 9 7950X 16-Core Processor"}}}, ""},
		{wire.Facts{Platform: "linux"}, ""},
	}
	for _, c := range cases {
		if got := Backend(c.f); got != c.want {
			t.Errorf("Backend(%+v) = %q", c.f, got)
		}
	}
}

func TestCollectOnThisHost(t *testing.T) {
	f := Collect(context.Background(), "", t.TempDir())
	if f.Platform != runtime.GOOS || f.Arch != runtime.GOARCH || f.Hostname == "" {
		t.Fatalf("facts %+v", f)
	}
	if f.MemoryBytes == 0 || f.DiskFreeBytes == 0 {
		t.Fatalf("memory or disk probe failed: %+v", f)
	}
}

func TestWithheldGPUsKeepTheirPhysicalIndex(t *testing.T) {
	kept := visible([]wire.GPU{{Index: 0, Name: "a"}, {Index: 1, Name: "b"}, {Index: 2, Name: "c"}}, "2", true)
	if len(kept) != 1 || kept[0].Index != 2 {
		t.Fatalf("kept %+v", kept)
	}
}

func TestFailedGPUProbeKeepsTheLastGoodList(t *testing.T) {
	good := []wire.GPU{{Index: 0, UUID: "GPU-1f2e3d4c", Name: "NVIDIA H200", MemoryBytes: 141 << 30}}
	fail := false
	c := &Collector{probe: func(context.Context, uint64) ([]wire.GPU, error) {
		if fail {
			return nil, errors.New("nvidia-smi: signal: killed")
		}
		return good, nil
	}}
	if g := c.gpuList(context.Background(), 0); !slices.Equal(g, good) {
		t.Fatalf("first probe %+v", g)
	}
	fail = true
	if g := c.gpuList(context.Background(), 0); !slices.Equal(g, good) {
		t.Fatalf("a failed probe reported %+v instead of the last good list", g)
	}
}
