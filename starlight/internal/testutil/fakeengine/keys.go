// Package fakeengine imitates `tensorfold serve` closely enough to test Starlight without a GPU or model.
package fakeengine

import (
	"bufio"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
)

// Key is one configured credential and the routes it reaches.
type Key struct {
	Label  string
	Scopes []string
	Secret string
}

func (k Key) full() bool {
	return len(k.Scopes) == 0 || (len(k.Scopes) == 1 && k.Scopes[0] == "admin")
}

func (k Key) has(scope string) bool {
	if k.full() {
		return true
	}
	for _, s := range k.Scopes {
		if s == scope {
			return true
		}
	}
	return false
}

// ParseKeyFile reads the engine's key-file format and refuses files other accounts can read, as the engine does.
func ParseKeyFile(path string) ([]Key, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if runtime.GOOS != "windows" && fi.Mode().Perm()&0o077 != 0 {
		return nil, fmt.Errorf("api key file %s is readable by other users; chmod 600 it", path)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var keys []Key
	sc := bufio.NewScanner(f)
	n := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		n++
		label, secret, ok := strings.Cut(line, ": ")
		if !ok {
			keys = append(keys, Key{Label: fmt.Sprintf("file-%d", n), Secret: line})
			continue
		}
		k := Key{Label: label, Secret: strings.TrimSpace(secret)}
		if i := strings.IndexByte(label, '['); i >= 0 {
			if !strings.HasSuffix(label, "]") {
				return nil, fmt.Errorf("malformed scopes in %q", label)
			}
			k.Label = label[:i]
			for _, s := range strings.Split(label[i+1:len(label)-1], ",") {
				switch s = strings.TrimSpace(s); s {
				case "admin", "inference", "read":
					k.Scopes = append(k.Scopes, s)
				default:
					return nil, fmt.Errorf("unknown scope %q", s)
				}
			}
		}
		if k.Secret == "" {
			return nil, errors.New("empty key")
		}
		keys = append(keys, k)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(keys) == 0 {
		return nil, errors.New("api key file holds no keys")
	}
	return keys, nil
}

func (e *Engine) match(token string) (Key, bool) {
	sum := sha256.Sum256([]byte(token))
	found, hit := Key{}, false
	for _, k := range e.opts.Keys {
		ks := sha256.Sum256([]byte(k.Secret))
		if subtle.ConstantTimeCompare(sum[:], ks[:]) == 1 {
			found, hit = k, true
		}
	}
	return found, hit
}
