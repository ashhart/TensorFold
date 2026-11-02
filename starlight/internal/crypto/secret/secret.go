// Package secret mints and recognises Starlight's bearer tokens; only their SHA-256 is ever stored.
package secret

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base32"
	"hash/crc32"
	"strings"
)

// Kind names what a token authorises; the prefix makes leaked tokens recognisable to scanners.
type Kind string

const (
	APIKey     Kind = "slr"
	Access     Kind = "sla"
	Refresh    Kind = "slf"
	Task       Kind = "slt"
	Enrollment Kind = "sle"
	SCIM       Kind = "sls"
	AppKey     Kind = "slc"
	Device     Kind = "sld"
	Recovery   Kind = "slx"
	Session    Kind = "slz"
)

var known = map[Kind]bool{APIKey: true, Access: true, Refresh: true, Task: true, Enrollment: true, SCIM: true, AppKey: true, Device: true, Recovery: true, Session: true}

var enc = base32.NewEncoding("abcdefghijklmnopqrstuvwxyz234567").WithPadding(base32.NoPadding)

const (
	idBytes     = 10
	secretBytes = 25
	idLen       = 16
	secretLen   = 40
	sumLen      = 7
	tokenLen    = 4 + idLen + 1 + secretLen + sumLen
)

// Generate returns a new token of kind k and the hash to store for it.
func Generate(k Kind) (token string, hash []byte) {
	raw := make([]byte, idBytes+secretBytes)
	if _, err := rand.Read(raw); err != nil {
		panic("secret: crypto/rand failed: " + err.Error())
	}
	body := string(k) + "_" + enc.EncodeToString(raw[:idBytes]) + "_" + enc.EncodeToString(raw[idBytes:])
	token = body + checksum(body)
	return token, Hash(token)
}

func checksum(body string) string {
	sum := crc32.ChecksumIEEE([]byte(body))
	b := []byte{byte(sum >> 24), byte(sum >> 16), byte(sum >> 8), byte(sum)}
	return enc.EncodeToString(b)
}

// Hash is the lookup digest stored in the database.
func Hash(token string) []byte {
	sum := sha256.Sum256([]byte(token))
	return sum[:]
}

// Parse reports the token's kind when its shape and checksum are valid.
func Parse(token string) (Kind, bool) {
	if len(token) != tokenLen || token[3] != '_' || token[4+idLen] != '_' {
		return "", false
	}
	k := Kind(token[:3])
	if !known[k] {
		return "", false
	}
	for i := 4; i < len(token); i++ {
		c := token[i]
		if i == 4+idLen {
			continue
		}
		if !(c >= 'a' && c <= 'z' || c >= '2' && c <= '7') {
			return "", false
		}
	}
	body := token[:len(token)-sumLen]
	if subtle.ConstantTimeCompare([]byte(checksum(body)), []byte(token[len(token)-sumLen:])) != 1 {
		return "", false
	}
	return k, true
}

// PublicID is the non-secret prefix shown in consoles and audit records.
func PublicID(token string) string {
	if len(token) < 4+idLen {
		return ""
	}
	return token[:4+idLen]
}

// Looks reports whether s resembles any Starlight token, for the egress secret scanner.
func Looks(s string) bool {
	_, ok := Parse(s)
	return ok
}

const userCodeAlphabet = "BCDFGHJKLMNPQRSTVWXZ"

// UserCode returns an eight-letter device code in the form ABCD-EFGH.
func UserCode() string {
	b := make([]byte, 8)
	out := make([]byte, 0, 9)
	for i := 0; i < 8; i++ {
		for {
			if _, err := rand.Read(b[i : i+1]); err != nil {
				panic(err)
			}
			if b[i] < 240 {
				break
			}
		}
		if i == 4 {
			out = append(out, '-')
		}
		out = append(out, userCodeAlphabet[int(b[i])%len(userCodeAlphabet)])
	}
	return string(out)
}

// NormalizeUserCode uppercases and strips separators so typed codes match stored ones.
func NormalizeUserCode(s string) string {
	s = strings.ToUpper(strings.Map(func(r rune) rune {
		if r == '-' || r == ' ' {
			return -1
		}
		return r
	}, s))
	if len(s) != 8 {
		return ""
	}
	for _, c := range s {
		if !strings.ContainsRune(userCodeAlphabet, c) {
			return ""
		}
	}
	return s[:4] + "-" + s[4:]
}

// Random returns n bytes from crypto/rand.
func Random(n int) []byte {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("secret: crypto/rand failed: " + err.Error())
	}
	return b
}

// RandomString returns an unpadded base32 string with at least n bytes of entropy.
func RandomString(n int) string {
	return enc.EncodeToString(Random(n))
}

// Equal compares secrets in constant time.
func Equal(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}
