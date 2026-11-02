// Package keyring seals secrets at rest with AES-256-GCM data keys wrapped by a master key.
package keyring

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

const sealVersion = 1

// ErrUnknownKey means a sealed value names a data key this keyring cannot unwrap.
var ErrUnknownKey = errors.New("keyring: unknown data key")

// Master is a key-encryption key identified by a fingerprint of its bytes.
type Master struct {
	ID   string
	aead cipher.AEAD
	prk  []byte
}

// NewMaster validates a 32-byte key and derives its identifier.
func NewMaster(key []byte) (*Master, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("master key must be 32 bytes, got %d", len(key))
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(append([]byte("starlight-master-id:"), key...))
	prk, err := hkdf.Extract(sha256.New, key, nil)
	if err != nil {
		return nil, err
	}
	return &Master{ID: hex.EncodeToString(sum[:6]), aead: aead, prk: prk}, nil
}

// Derive returns n bytes, at most 8160, for label by HKDF-SHA256 from the master key: alike wherever it is loaded.
func (m *Master) Derive(label string, n int) []byte {
	out, err := hkdf.Expand(sha256.New, m.prk, label, n)
	if err != nil {
		panic(err)
	}
	return out
}

// LoadMaster reads a base64 master key from a value or a file path, preferring the value.
func LoadMaster(value, file string) (*Master, error) {
	raw := strings.TrimSpace(value)
	if raw == "" && file != "" {
		b, err := os.ReadFile(file)
		if err != nil {
			return nil, fmt.Errorf("master key file: %w", err)
		}
		raw = strings.TrimSpace(string(b))
	}
	if raw == "" {
		return nil, errors.New("no master key: set STARLIGHT_MASTER_KEY or master_key_file")
	}
	key, err := base64.StdEncoding.DecodeString(raw)
	if err != nil {
		return nil, fmt.Errorf("master key is not base64: %w", err)
	}
	return NewMaster(key)
}

// GenerateMasterKey returns a fresh base64 master key.
func GenerateMasterKey() string {
	k := make([]byte, 32)
	if _, err := rand.Read(k); err != nil {
		panic(err)
	}
	return base64.StdEncoding.EncodeToString(k)
}

// Keyring holds unwrapped data keys and the id of the one used for new seals.
type Keyring struct {
	mu      sync.RWMutex
	masters map[string]*Master
	primary *Master
	keys    map[string]cipher.AEAD
	active  string
}

// New builds a keyring; the first master wraps new data keys, the rest only unwrap.
func New(primary *Master, previous ...*Master) *Keyring {
	k := &Keyring{masters: map[string]*Master{primary.ID: primary}, primary: primary, keys: map[string]cipher.AEAD{}}
	for _, m := range previous {
		k.masters[m.ID] = m
	}
	return k
}

// Derive returns n bytes for label from the primary master key; see Master.Derive.
func (k *Keyring) Derive(label string, n int) []byte { return k.primary.Derive(label, n) }

// Load unwraps every stored data key and creates the first one when none exists.
func (k *Keyring) Load(ctx context.Context, q db.Querier) error {
	rows, err := q.Query(ctx, "SELECT id, wrapped, master_key_id, status FROM data_keys")
	if err != nil {
		return err
	}
	defer rows.Close()
	k.mu.Lock()
	defer k.mu.Unlock()
	for rows.Next() {
		var id, masterID, status string
		var wrapped []byte
		if err := rows.Scan(&id, &wrapped, &masterID, &status); err != nil {
			return err
		}
		m, ok := k.masters[masterID]
		if !ok {
			return fmt.Errorf("data key %s is wrapped by master %s, which is not loaded", id, masterID)
		}
		aead, err := unwrap(m, id, wrapped)
		if err != nil {
			return fmt.Errorf("data key %s: %w", id, err)
		}
		k.keys[id] = aead
		if status == "active" {
			k.active = id
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if k.active == "" {
		return errors.New("keyring: no active data key; run `starlight init` or `starlight keys rotate`")
	}
	return nil
}

// Rotate creates a new active data key and retires the previous one for new seals.
func (k *Keyring) Rotate(ctx context.Context, q db.Querier) (string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	idBytes := make([]byte, 6)
	if _, err := rand.Read(idBytes); err != nil {
		return "", err
	}
	id := hex.EncodeToString(idBytes)
	wrapped, err := k.wrap(id, raw)
	if err != nil {
		return "", err
	}
	if _, err := q.Exec(ctx, "UPDATE data_keys SET status = 'retired' WHERE status = 'active'"); err != nil {
		return "", err
	}
	if _, err := q.Exec(ctx, "INSERT INTO data_keys (id, wrapped, master_key_id, status) VALUES ($1, $2, $3, 'active')", id, wrapped, k.primary.ID); err != nil {
		return "", err
	}
	aead, err := newAEAD(raw)
	if err != nil {
		return "", err
	}
	k.mu.Lock()
	k.keys[id] = aead
	k.active = id
	k.mu.Unlock()
	return id, nil
}

// Rewrap moves every data key wrapped by a previous master under the primary, so old master files can be retired.
func (k *Keyring) Rewrap(ctx context.Context, q db.Querier) (int, error) {
	rows, err := q.Query(ctx, "SELECT id, wrapped, master_key_id FROM data_keys WHERE master_key_id <> $1", k.primary.ID)
	if err != nil {
		return 0, err
	}
	type held struct {
		id, master string
		wrapped    []byte
	}
	var all []held
	for rows.Next() {
		var h held
		if err := rows.Scan(&h.id, &h.wrapped, &h.master); err != nil {
			rows.Close()
			return 0, err
		}
		all = append(all, h)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}
	for i, h := range all {
		m, ok := k.masters[h.master]
		if !ok {
			return i, fmt.Errorf("data key %s is wrapped by master %s, which is not loaded", h.id, h.master)
		}
		raw, err := unwrapRaw(m, h.id, h.wrapped)
		if err != nil {
			return i, fmt.Errorf("data key %s: %w", h.id, err)
		}
		wrapped, err := k.wrap(h.id, raw)
		if err != nil {
			return i, err
		}
		if _, err := q.Exec(ctx, "UPDATE data_keys SET wrapped = $2, master_key_id = $3 WHERE id = $1 AND master_key_id = $4",
			h.id, wrapped, k.primary.ID, h.master); err != nil {
			return i, err
		}
	}
	return len(all), nil
}

func (k *Keyring) wrap(id string, raw []byte) ([]byte, error) {
	nonce := make([]byte, k.primary.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	return k.primary.aead.Seal(nonce, nonce, raw, []byte("starlight-dek:"+id)), nil
}

func unwrap(m *Master, id string, wrapped []byte) (cipher.AEAD, error) {
	raw, err := unwrapRaw(m, id, wrapped)
	if err != nil {
		return nil, err
	}
	return newAEAD(raw)
}

func unwrapRaw(m *Master, id string, wrapped []byte) ([]byte, error) {
	ns := m.aead.NonceSize()
	if len(wrapped) < ns {
		return nil, errors.New("wrapped key too short")
	}
	raw, err := m.aead.Open(nil, wrapped[:ns], wrapped[ns:], []byte("starlight-dek:"+id))
	if err != nil {
		return nil, errors.New("cannot unwrap with this master key")
	}
	return raw, nil
}

func newAEAD(raw []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(raw)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

// Seal encrypts plaintext bound to aad, which names the row and column it belongs to.
func (k *Keyring) Seal(plaintext []byte, aad string) ([]byte, error) {
	k.mu.RLock()
	id, aead := k.active, k.keys[k.active]
	k.mu.RUnlock()
	if aead == nil {
		return nil, errors.New("keyring: not loaded")
	}
	out := make([]byte, 0, 2+len(id)+aead.NonceSize()+len(plaintext)+aead.Overhead())
	out = append(out, sealVersion, byte(len(id)))
	out = append(out, id...)
	nonce := make([]byte, aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	out = append(out, nonce...)
	return aead.Seal(out, nonce, plaintext, []byte(aad)), nil
}

// Open decrypts a sealed value and fails if aad differs from the one it was sealed with.
func (k *Keyring) Open(sealed []byte, aad string) ([]byte, error) {
	if len(sealed) < 2 || sealed[0] != sealVersion {
		return nil, errors.New("keyring: unsupported sealed value")
	}
	n := int(sealed[1])
	if len(sealed) < 2+n {
		return nil, errors.New("keyring: truncated sealed value")
	}
	id := string(sealed[2 : 2+n])
	k.mu.RLock()
	aead := k.keys[id]
	k.mu.RUnlock()
	if aead == nil {
		return nil, ErrUnknownKey
	}
	rest := sealed[2+n:]
	ns := aead.NonceSize()
	if len(rest) < ns+aead.Overhead() {
		return nil, errors.New("keyring: truncated sealed value")
	}
	out, err := aead.Open(nil, rest[:ns], rest[ns:], []byte(aad))
	if err != nil {
		return nil, errors.New("keyring: sealed value failed authentication")
	}
	return out, nil
}

// KeyID returns the data key a sealed value uses, for rewrap planning.
func KeyID(sealed []byte) string {
	if len(sealed) < 2 || len(sealed) < 2+int(sealed[1]) {
		return ""
	}
	return string(sealed[2 : 2+int(sealed[1])])
}

// Active returns the data key id used for new seals.
func (k *Keyring) Active() string {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.active
}

// ForTest builds a loaded in-memory keyring with one data key.
func ForTest() *Keyring {
	m, _ := NewMaster(make([]byte, 32))
	k := New(m)
	raw := make([]byte, 32)
	_, _ = rand.Read(raw)
	aead, _ := newAEAD(raw)
	k.keys["test"] = aead
	k.active = "test"
	return k
}
