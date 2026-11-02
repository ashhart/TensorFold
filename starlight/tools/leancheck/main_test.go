package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModuleIsLean(t *testing.T) {
	findings, err := Check(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range findings {
		t.Errorf("%s:%d: %s", f.Path, f.Line, f.Rule)
	}
}

func TestDetectsViolations(t *testing.T) {
	dir := t.TempDir()
	src := "package x\n\n// one\n// two\nfunc a() {}\n/* block */\nvar s = \"/* not a comment */\"\n"
	if err := os.WriteFile(filepath.Join(dir, "x.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	findings, err := Check(dir)
	if err != nil {
		t.Fatal(err)
	}
	rules := map[string]int{}
	for _, f := range findings {
		rules[f.Rule]++
	}
	if rules["multi-line comment"] != 1 || rules["block comment"] != 1 {
		t.Fatalf("findings = %v", findings)
	}
}
