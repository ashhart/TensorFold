package tray

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestEmbeddedIconIsSquarePNG(t *testing.T) {
	w, h, err := PNGSize(iconPNG)
	if err != nil || w != 44 || h != 44 {
		t.Fatalf("icon is %dx%d, %v", w, h, err)
	}
}

func TestICOWrapsPNG(t *testing.T) {
	ico, err := ICO(iconPNG)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ico[:6], []byte{0, 0, 1, 0, 1, 0}) || ico[6] != 44 || ico[7] != 44 {
		t.Fatalf("header % x", ico[:8])
	}
	if size := binary.LittleEndian.Uint32(ico[14:18]); int(size) != len(iconPNG) {
		t.Fatalf("entry size %d, png %d", size, len(iconPNG))
	}
	if off := binary.LittleEndian.Uint32(ico[18:22]); off != 22 || !bytes.Equal(ico[22:], iconPNG) {
		t.Fatalf("payload offset %d", off)
	}
	if _, err := ICO([]byte("not a png")); err == nil {
		t.Fatal("wrapped a non-PNG")
	}
}
