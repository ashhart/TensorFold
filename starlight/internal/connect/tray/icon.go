// Package tray is Connect's menu-bar and system-tray app.
package tray

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"errors"
	"runtime"
)

//go:embed assets/icon.png
var iconPNG []byte

var pngSignature = []byte("\x89PNG\r\n\x1a\n")

// PNGSize reads the width and height from a PNG's IHDR chunk.
func PNGSize(png []byte) (int, int, error) {
	if len(png) < 24 || !bytes.Equal(png[:8], pngSignature) || string(png[12:16]) != "IHDR" {
		return 0, 0, errors.New("not a PNG image")
	}
	return int(binary.BigEndian.Uint32(png[16:20])), int(binary.BigEndian.Uint32(png[20:24])), nil
}

// ICO wraps a PNG in a single-image ICO container, the format Windows tray icons take.
func ICO(png []byte) ([]byte, error) {
	w, h, err := PNGSize(png)
	if err != nil {
		return nil, err
	}
	if w > 256 || h > 256 {
		return nil, errors.New("ICO images are at most 256 pixels square")
	}
	dim := func(n int) byte {
		if n == 256 {
			return 0
		}
		return byte(n)
	}
	var b bytes.Buffer
	b.Write([]byte{0, 0, 1, 0, 1, 0})
	b.Write([]byte{dim(w), dim(h), 0, 0})
	binary.Write(&b, binary.LittleEndian, uint16(1))
	binary.Write(&b, binary.LittleEndian, uint16(32))
	binary.Write(&b, binary.LittleEndian, uint32(len(png)))
	binary.Write(&b, binary.LittleEndian, uint32(22))
	b.Write(png)
	return b.Bytes(), nil
}

// Icon returns the tray icon in the platform's format.
func Icon() []byte {
	if runtime.GOOS == "windows" {
		if ico, err := ICO(iconPNG); err == nil {
			return ico
		}
	}
	return iconPNG
}
