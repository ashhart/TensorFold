package facts

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// rocminfo returns a testdata sample: a fictional host in rocminfo's layout, with its trailing blanks stripped.
func rocminfo(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "rocminfo", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func indexes(gpus []wire.GPU) string {
	var s []string
	for _, g := range gpus {
		s = append(s, strconv.Itoa(g.Index))
	}
	return strings.Join(s, ",")
}

func TestParseRocminfoSamples(t *testing.T) {
	ws, mi, apu := rocminfo(t, "radeon-pro-workstation.txt"), rocminfo(t, "instinct-mi300x-node.txt"), rocminfo(t, "strix-halo-apu.txt")
	w7900 := wire.GPU{Index: 0, Name: "AMD Radeon PRO W7900", MemoryBytes: 50315264 << 10, UUID: "GPU-1b2c3d4e5f6a7b8c"}
	w7800 := wire.GPU{Index: 1, Name: "AMD Radeon PRO W7800", MemoryBytes: 33538048 << 10, UUID: "GPU-9a8b7c6d5e4f3a2b"}
	unnamed := w7800
	unnamed.Name = "AMD gfx1100"
	var instinct, shared []wire.GPU
	for i, id := range []string{"0a1b2c3d4e5f6071", "1a2b3c4d5e6f7081", "2a3b4c5d6e7f8091", "3a4b5c6d7e8f90a1",
		"4a5b6c7d8e9fa0b1", "5a6b7c8d9eafb0c1", "6a7b8c9daebfc0d1", "7a8b9cadbecfd0e1"} {
		g := wire.GPU{Index: i, Name: "AMD Instinct MI300X", MemoryBytes: 201310208 << 10, UUID: "GPU-" + id}
		instinct = append(instinct, g)
		g.MemoryBytes = 0
		shared = append(shared, g)
	}
	strix := []wire.GPU{{Index: 0, Name: "AMD Radeon 8060S Graphics"}}
	vega := wire.GPU{Name: "AMD Vega 20", MemoryBytes: 16760832 << 10}
	vega1 := vega
	vega1.Index = 1
	for _, c := range []struct {
		name, text string
		want       []wire.GPU
	}{
		{"two-GPU workstation", ws, []wire.GPU{w7900, w7800}},
		{"eight-GPU Instinct node", mi, instinct},
		{"APU", apu, strix},
		{"APU known by its name alone", strings.Replace(apu, "       APU", "", 1), strix},
		{"APUs rocminfo marks, as on MI300A", strings.ReplaceAll(mi, "Memory Properties:\n", "Memory Properties:       APU\n"), shared},
		{"no marketing name", strings.Replace(ws, "AMD Radeon PRO W7800", "", 1), []wire.GPU{w7900, unnamed}},
		{"no Uuid lines", rocminfo(t, "no-uuid.txt"), []wire.GPU{vega, vega1}},
		{"CPU only", rocminfo(t, "cpu-only.txt"), nil},
	} {
		// rocminfo pads every label and value with blanks, so each sample is also read padded
		for _, text := range []string{c.text, strings.ReplaceAll(c.text, "\n", "      \n")} {
			got := hipVisible(parseRocminfo(text), "")
			if !slices.Equal(got, c.want) {
				t.Errorf("%s: parsed %+v, want %+v", c.name, got, c.want)
			}
			if b := Backend(wire.Facts{Platform: "linux", GPUs: got}); (b == "hip") != (len(got) > 0) {
				t.Errorf("%s: backend %q for %d GPUs", c.name, b, len(got))
			}
		}
	}
}

func TestHIPVisibleFollowsClrRules(t *testing.T) {
	agents := parseRocminfo(rocminfo(t, "instinct-mi300x-node.txt"))
	all := "0,1,2,3,4,5,6,7"
	for _, c := range []struct {
		env  []string
		want string
	}{
		{nil, all},
		{[]string{"ROCR_VISIBLE_DEVICES=1"}, all},
		{[]string{"HIP_VISIBLE_DEVICES=2,0"}, "2,0"},
		{[]string{"HIP_VISIBLE_DEVICES=0,9,2"}, "0"},
		{[]string{"HIP_VISIBLE_DEVICES=2,0,2,1"}, "2,0,1"},
		{[]string{"HIP_VISIBLE_DEVICES=1,x,0"}, "1"},
		{[]string{"HIP_VISIBLE_DEVICES=1,,0"}, "1"},
		{[]string{"HIP_VISIBLE_DEVICES=3,"}, "3"},
		{[]string{"HIP_VISIBLE_DEVICES=1, 2"}, "1"},
		{[]string{"HIP_VISIBLE_DEVICES= 1"}, ""},
		{[]string{"HIP_VISIBLE_DEVICES=01"}, ""},
		{[]string{"HIP_VISIBLE_DEVICES=+1"}, ""},
		{[]string{"HIP_VISIBLE_DEVICES=-1"}, ""},
		{[]string{"HIP_VISIBLE_DEVICES=GPU-3a4b5c6d7e8f90a1"}, "3"},
		{[]string{"HIP_VISIBLE_DEVICES=GPU-5a6b,0"}, "5,0"},
		{[]string{"HIP_VISIBLE_DEVICES=1,GPU-1a2b,4"}, "1,4"},
		{[]string{"HIP_VISIBLE_DEVICES=2,GPU-ffff,0"}, "2"},
		{[]string{"HIP_VISIBLE_DEVICES=GPU-5A6B"}, ""},
		{[]string{`HIP_VISIBLE_DEVICES="3"`}, "3"},
		{[]string{`HIP_VISIBLE_DEVICES= "3,4" `}, "3,4"},
		{[]string{"HIP_VISIBLE_DEVICES=6", "CUDA_VISIBLE_DEVICES=5"}, "6"},
		{[]string{`HIP_VISIBLE_DEVICES=""`, "CUDA_VISIBLE_DEVICES=4"}, "4"},
		{[]string{"HIP_VISIBLE_DEVICES=", "CUDA_VISIBLE_DEVICES=4"}, ""},
		{[]string{"CUDA_VISIBLE_DEVICES=5,7"}, "5,7"},
		{[]string{"CUDA_VISIBLE_DEVICES="}, ""},
		{[]string{`CUDA_VISIBLE_DEVICES=""`}, all},
	} {
		lookup := func(k string) (string, bool) {
			for _, kv := range c.env {
				if name, v, _ := strings.Cut(kv, "="); name == k {
					return v, true
				}
			}
			return "", false
		}
		if got := indexes(hipVisible(agents, hipList(lookup))); got != c.want {
			t.Errorf("%q: kept GPUs %q, want %q", c.env, got, c.want)
		}
	}
}

func TestUUIDsThatPinNothingAreNotReported(t *testing.T) {
	// HIP still finds GPU- entries in the Uuid text, but a shared or placeholder Uuid would pin the wrong GPU
	twins := strings.Replace(rocminfo(t, "radeon-pro-workstation.txt"), "GPU-9a8b7c6d5e4f3a2b", "GPU-1b2c3d4e5f6a7b8c", 1)
	apu, old := rocminfo(t, "strix-halo-apu.txt"), rocminfo(t, "no-uuid.txt")
	for _, c := range []struct{ name, text, list, want string }{
		{"two agents print one Uuid", twins, "", "0,1"},
		{"two agents print one Uuid", twins, "GPU-1b2c3d4e5f6a7b8c", "0"},
		{"GPU-XX", apu, "GPU-XX", "0"},
		{"no Uuid lines", old, "1,0", "1,0"},
		{"no Uuid lines", old, "1,GPU-0", "1"},
	} {
		got := hipVisible(parseRocminfo(c.text), c.list)
		if indexes(got) != c.want {
			t.Errorf("%s, list %q: kept GPUs %q, want %q", c.name, c.list, indexes(got), c.want)
		}
		for _, g := range got {
			if g.UUID != "" {
				t.Errorf("%s: GPU %d reports UUID %q", c.name, g.Index, g.UUID)
			}
		}
	}
}

func TestAMDGPUsRunsRocminfoUnderTheAgentsHIPList(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in rocminfo is a shell script")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "rocminfo.txt"), []byte(rocminfo(t, "radeon-pro-workstation.txt")), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "rocminfo"), []byte("#!/bin/sh\nexec cat \"$0.txt\"\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HIP_VISIBLE_DEVICES", "1")
	want := wire.GPU{Index: 1, Name: "AMD Radeon PRO W7800", MemoryBytes: 33538048 << 10, UUID: "GPU-9a8b7c6d5e4f3a2b"}
	if got, err := amdGPUs(context.Background()); err != nil || len(got) != 1 || got[0] != want {
		t.Fatalf("amdGPUs = %+v, %v, want [%+v]", got, err, want)
	}
}
