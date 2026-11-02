// Package permit signs the single-use authority a gateway hands a node for one inference request.
package permit

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

const (
	version = "v1"
	domain  = "starlight-permit-v1\n"
	// MaxLifetime bounds how long a permit may be honoured after it is issued.
	MaxLifetime = 2 * time.Minute
	skew        = 5 * time.Second
)

// Claims bind a permit to one node, replica, request body and deadline.
type Claims struct {
	RequestID  string `json:"rid"`
	Node       string `json:"nid"`
	Replica    string `json:"rep"`
	Deployment string `json:"dep"`
	Revision   int    `json:"rev"`
	Method     string `json:"m"`
	Path       string `json:"p"`
	BodySHA256 string `json:"sha"`
	IssuedAt   int64  `json:"iat"`
	Expires    int64  `json:"exp"`
	Subject    string `json:"sub"`
	Priority   string `json:"pri"`
	MaxTokens  int    `json:"mx,omitempty"`
}

// Signer issues permits with the active key.
type Signer struct {
	KID string
	Key ed25519.PrivateKey
}

// Sign encodes and signs claims.
func (s Signer) Sign(c Claims) (string, error) {
	if s.KID == "" || strings.ContainsAny(s.KID, ".") || len(s.Key) != ed25519.PrivateKeySize {
		return "", errors.New("permit: signer not configured")
	}
	body, err := json.Marshal(c)
	if err != nil {
		return "", err
	}
	payload := base64.RawURLEncoding.EncodeToString(body)
	signed := version + "." + s.KID + "." + payload
	sig := ed25519.Sign(s.Key, []byte(domain+signed))
	return signed + "." + base64.RawURLEncoding.EncodeToString(sig), nil
}

// Errors returned by Verify; callers map them to 401 or 403.
var (
	ErrMalformed = errors.New("permit: malformed")
	ErrKey       = errors.New("permit: unknown signing key")
	ErrSignature = errors.New("permit: bad signature")
	ErrExpired   = errors.New("permit: expired")
)

// Verifier checks permits against the published public keys.
type Verifier struct {
	mu   sync.RWMutex
	keys map[string]ed25519.PublicKey
}

// NewVerifier starts with the given keys.
func NewVerifier(keys map[string]ed25519.PublicKey) *Verifier {
	v := &Verifier{}
	v.SetKeys(keys)
	return v
}

// SetKeys replaces the trusted key set, as the control plane rotates keys.
func (v *Verifier) SetKeys(keys map[string]ed25519.PublicKey) {
	cp := make(map[string]ed25519.PublicKey, len(keys))
	for k, pk := range keys {
		if len(pk) == ed25519.PublicKeySize {
			cp[k] = pk
		}
	}
	v.mu.Lock()
	v.keys = cp
	v.mu.Unlock()
}

// Verify checks signature and time bounds and returns the claims.
func (v *Verifier) Verify(token string, now time.Time) (Claims, error) {
	var c Claims
	parts := strings.Split(token, ".")
	if len(parts) != 4 || parts[0] != version {
		return c, ErrMalformed
	}
	v.mu.RLock()
	pk := v.keys[parts[1]]
	v.mu.RUnlock()
	if pk == nil {
		return c, ErrKey
	}
	sig, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil || len(sig) != ed25519.SignatureSize {
		return c, ErrMalformed
	}
	signed := parts[0] + "." + parts[1] + "." + parts[2]
	if !ed25519.Verify(pk, []byte(domain+signed), sig) {
		return c, ErrSignature
	}
	body, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return c, ErrMalformed
	}
	dec := json.NewDecoder(strings.NewReader(string(body)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return c, ErrMalformed
	}
	iat, exp := time.UnixMilli(c.IssuedAt), time.UnixMilli(c.Expires)
	if exp.Sub(iat) > MaxLifetime || exp.Before(iat) {
		return c, fmt.Errorf("%w: lifetime out of bounds", ErrMalformed)
	}
	if now.Add(skew).Before(iat) || now.After(exp.Add(skew)) {
		return c, ErrExpired
	}
	if c.RequestID == "" || c.Node == "" || c.Replica == "" || len(c.BodySHA256) != 64 {
		return c, ErrMalformed
	}
	return c, nil
}

// Replay remembers request IDs until their permits expire so each permit is used once.
type Replay struct {
	mu   sync.Mutex
	seen map[string]time.Time
	next time.Time
}

// NewReplay returns an empty replay cache.
func NewReplay() *Replay { return &Replay{seen: map[string]time.Time{}} }

// Use records rid and reports false when it was already used.
func (r *Replay) Use(rid string, expires, now time.Time) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if now.After(r.next) {
		for k, t := range r.seen {
			if now.After(t.Add(skew)) {
				delete(r.seen, k)
			}
		}
		r.next = now.Add(10 * time.Second)
	}
	if _, dup := r.seen[rid]; dup {
		return false
	}
	r.seen[rid] = expires
	return true
}
