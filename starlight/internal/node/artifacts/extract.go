package artifacts

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// Limits caps what one archive may write.
type Limits struct {
	MaxBytes int64
	MaxFiles int
}

// Extract unpacks a tar or tar.gz into dest, allowing only directories and regular files inside dest.
func Extract(archive, dest string, lim Limits) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)
	var r io.Reader = br
	if magic, err := br.Peek(2); err == nil && magic[0] == 0x1f && magic[1] == 0x8b {
		gz, err := gzip.NewReader(br)
		if err != nil {
			return fmt.Errorf("archive: %w", err)
		}
		defer gz.Close()
		r = gz
	}
	return extractTar(tar.NewReader(r), dest, lim)
}

func extractTar(tr *tar.Reader, dest string, lim Limits) error {
	var written int64
	files := 0
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("archive: %w", err)
		}
		target, err := entryPath(dest, h.Name)
		if err != nil {
			return err
		}
		if target == "" {
			continue
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o700); err != nil {
				return err
			}
		case tar.TypeReg, '\x00':
			files++
			if files > lim.MaxFiles {
				return fmt.Errorf("archive holds more than %d files", lim.MaxFiles)
			}
			if h.Size < 0 || written+h.Size > lim.MaxBytes {
				return fmt.Errorf("archive expands beyond the %d byte cap", lim.MaxBytes)
			}
			if err := writeFile(target, tr, h.Size); err != nil {
				return err
			}
			written += h.Size
		default:
			return fmt.Errorf("archive entry %q is not a directory or regular file; links and devices are refused", h.Name)
		}
	}
}

// entryPath maps an archive name into dest, refusing absolute, parent-relative and backslash names.
func entryPath(dest, name string) (string, error) {
	if strings.ContainsAny(name, "\x00\\") {
		return "", fmt.Errorf("archive entry %q has a forbidden character", name)
	}
	if strings.HasPrefix(name, "/") || filepath.VolumeName(name) != "" {
		return "", fmt.Errorf("archive entry %q is absolute", name)
	}
	clean := path.Clean(name)
	if clean == "." {
		return "", nil
	}
	if clean == ".." || strings.HasPrefix(clean, "../") {
		return "", fmt.Errorf("archive entry %q escapes the destination", name)
	}
	for _, part := range strings.Split(clean, "/") {
		if part == ".." {
			return "", fmt.Errorf("archive entry %q escapes the destination", name)
		}
	}
	target := filepath.Join(dest, filepath.FromSlash(clean))
	rel, err := filepath.Rel(dest, target)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("archive entry %q escapes the destination", name)
	}
	return target, nil
}

func writeFile(target string, r io.Reader, size int64) error {
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return err
	}
	f, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("archive: %w", err)
	}
	n, err := io.CopyN(f, r, size)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("archive: %w", err)
	}
	if n != size {
		return errors.New("archive: short file")
	}
	return nil
}
