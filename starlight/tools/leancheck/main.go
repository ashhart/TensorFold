// Command leancheck enforces the repository's lean rules on Go sources: small files and one-line comments.
package main

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

const (
	maxLines   = 600
	maxComment = 120
)

// Finding is one rule violation.
type Finding struct {
	Path string
	Line int
	Rule string
}

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	findings, err := Check(root)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	for _, f := range findings {
		fmt.Printf("%s:%d: %s\n", f.Path, f.Line, f.Rule)
	}
	if len(findings) > 0 {
		fmt.Printf("%d problems\n", len(findings))
		os.Exit(1)
	}
}

// Check walks root and reports every violation in .go files.
func Check(root string) ([]Finding, error) {
	var out []Finding
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "vendor" || d.Name() == "testdata" || strings.HasPrefix(d.Name(), ".")) && path != root {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		fs, err := checkFile(path)
		out = append(out, fs...)
		return err
	})
	return out, err
}

func checkFile(path string) ([]Finding, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Finding
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	n, prevComment := 0, false
	for sc.Scan() {
		n++
		line := strings.TrimSpace(sc.Text())
		code := line
		if i := strings.Index(code, "//"); i >= 0 && !inString(code, "//") {
			code = code[:i]
		}
		if strings.Contains(code, "/*") && !inString(code, "/*") {
			out = append(out, Finding{path, n, "block comment"})
		}
		isComment := strings.HasPrefix(line, "//") && !strings.HasPrefix(line, "//go:") && !strings.HasPrefix(line, "//nolint")
		if isComment {
			if prevComment {
				out = append(out, Finding{path, n, "multi-line comment"})
			}
			if len(sc.Text()) > maxComment {
				out = append(out, Finding{path, n, fmt.Sprintf("comment longer than %d columns", maxComment)})
			}
		}
		if i := strings.Index(line, " // "); i > 0 && !inString(line, " // ") && len(sc.Text()) > maxComment {
			out = append(out, Finding{path, n, fmt.Sprintf("trailing comment line longer than %d columns", maxComment)})
		}
		prevComment = isComment
	}
	if n > maxLines {
		out = append(out, Finding{path, n, fmt.Sprintf("file has %d lines, limit %d", n, maxLines)})
	}
	return out, sc.Err()
}

// inString reports whether the first occurrence of tok sits inside a quoted or backquoted string.
func inString(line, tok string) bool {
	idx := strings.Index(line, tok)
	var quote rune
	for i, c := range line {
		if i >= idx {
			return quote != 0
		}
		switch {
		case quote == 0 && (c == '"' || c == '`' || c == '\''):
			quote = c
		case quote != 0 && c == quote && (i == 0 || line[i-1] != '\\'):
			quote = 0
		}
	}
	return false
}
