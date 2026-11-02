package service

import (
	"os"
	"path/filepath"
	"testing"
)

// The reference files under deploy/node are exactly what `service install` writes with the documented defaults.
func TestDeployFilesMatchInstaller(t *testing.T) {
	darwinState := "/Library/Application Support/TensorFold/Starlight Node"
	plist, err := Plist(Options{Binary: "/usr/local/bin/starlight-node", User: "_tensorfold", StateDir: darwinState,
		Args: RunArgs(darwinState, []string{"--engine", "/opt/homebrew/bin/tensorfold", "--prevent-sleep"})})
	if err != nil {
		t.Fatal(err)
	}
	unit, err := Unit(Options{Binary: "/usr/local/bin/starlight-node", User: "starlight-node", StateDir: "/var/lib/starlight-node",
		Args: RunArgs("/var/lib/starlight-node", []string{"--engine", "/opt/tensorfold/bin/tensorfold-native", "--hf-home", "/var/lib/starlight-node/hf"})})
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]string{"dev.tensorfold.starlight-node.plist": plist, "starlight-node.service": unit}
	for name, want := range files {
		path := filepath.Join("..", "..", "..", "deploy", "node", name)
		if os.Getenv("STARLIGHT_UPDATE_GOLDEN") == "1" {
			if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != want {
			t.Errorf("%s differs from what the installer writes; regenerate with STARLIGHT_UPDATE_GOLDEN=1", name)
		}
	}
}
