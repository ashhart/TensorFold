// Package ids issues time-ordered UUIDv7 identifiers.
package ids

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"sync"
	"time"
)

var (
	mu       sync.Mutex
	lastMs   uint64
	sequence uint16
)

// New returns a UUIDv7 string; IDs from one process are strictly increasing.
func New() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("ids: crypto/rand failed: " + err.Error())
	}
	mu.Lock()
	ms := uint64(time.Now().UnixMilli())
	if ms <= lastMs {
		sequence++
		if sequence == 0x0fff {
			lastMs++
			sequence = 0
		}
		ms = lastMs
	} else {
		lastMs = ms
		sequence = binary.BigEndian.Uint16(b[6:8]) & 0x07ff
	}
	seq := sequence
	mu.Unlock()
	b[0] = byte(ms >> 40)
	b[1] = byte(ms >> 32)
	b[2] = byte(ms >> 24)
	b[3] = byte(ms >> 16)
	b[4] = byte(ms >> 8)
	b[5] = byte(ms)
	b[6] = 0x70 | byte(seq>>8)
	b[7] = byte(seq)
	b[8] = b[8]&0x3f | 0x80
	return format(b)
}

func format(b [16]byte) string {
	var out [36]byte
	hex.Encode(out[0:8], b[0:4])
	out[8] = '-'
	hex.Encode(out[9:13], b[4:6])
	out[13] = '-'
	hex.Encode(out[14:18], b[6:8])
	out[18] = '-'
	hex.Encode(out[19:23], b[8:10])
	out[23] = '-'
	hex.Encode(out[24:36], b[10:16])
	return string(out[:])
}

// ErrInvalid reports a malformed identifier.
var ErrInvalid = errors.New("invalid id")

// Valid reports whether s is a canonical lowercase UUID string.
func Valid(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i := 0; i < 36; i++ {
		c := s[i]
		switch i {
		case 8, 13, 18, 23:
			if c != '-' {
				return false
			}
		default:
			if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
				return false
			}
		}
	}
	return true
}

// Parse validates s and returns it unchanged, so callers can reject bad path parameters early.
func Parse(s string) (string, error) {
	if !Valid(s) {
		return "", ErrInvalid
	}
	return s, nil
}
