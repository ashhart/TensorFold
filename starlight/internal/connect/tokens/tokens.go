// Package tokens keeps Connect's access token fresh, refreshing once for every waiting caller and process.
package tokens

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fsutil"
	"github.com/ashhart/TensorFold/starlight/internal/connect/oauth"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

// Key is the credential-store entry that holds the token set.
const Key = "tokens"

// early refreshes an access token this long before it expires.
const early = 60 * time.Second

// ErrSignedOut means there is no usable sign-in; the user must run login.
var ErrSignedOut = errors.New("TensorFold Connect is signed out; run: tensorfold-connect login")

// Set is the stored sign-in for one server.
type Set struct {
	Server       string    `json:"server"`
	AccessToken  string    `json:"access_token"`
	Expiry       time.Time `json:"expiry"`
	RefreshToken string    `json:"refresh_token"`
	Scope        string    `json:"scope"`
}

// Manager hands out access tokens and refreshes them under a lock shared with other Connect processes.
type Manager struct {
	store  store.Store
	lock   string
	client *http.Client
	doc    discovery.Document
	server string
	now    func() time.Time
	mu     sync.Mutex
	cur    *Set
}

// New returns a manager for doc's server; lockPath serialises refreshes across processes.
func New(st store.Store, lockPath string, client *http.Client, doc discovery.Document) *Manager {
	return &Manager{store: st, lock: lockPath, client: client, doc: doc, server: doc.Issuer, now: time.Now}
}

func (m *Manager) load() (*Set, error) {
	data, err := m.store.Get(Key)
	if errors.Is(err, store.ErrNotFound) {
		return nil, ErrSignedOut
	}
	if err != nil {
		return nil, err
	}
	var s Set
	if err := json.Unmarshal(data, &s); err != nil || s.Server != m.server || s.AccessToken == "" {
		return nil, ErrSignedOut
	}
	return &s, nil
}

func (m *Manager) fresh(s *Set) bool {
	return s != nil && m.now().Add(early).Before(s.Expiry)
}

func (m *Manager) persist(tr discovery.TokenResponse, previousRefresh string) (*Set, error) {
	s := &Set{Server: m.server, AccessToken: tr.AccessToken, RefreshToken: tr.RefreshToken, Scope: tr.Scope,
		Expiry: m.now().Add(time.Duration(tr.ExpiresIn) * time.Second)}
	if s.RefreshToken == "" {
		s.RefreshToken = previousRefresh
	}
	if tr.ExpiresIn <= 0 {
		s.Expiry = m.now().Add(5 * time.Minute)
	}
	data, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	if err := m.store.Set(Key, data); err != nil {
		return nil, err
	}
	return s, nil
}

// Save stores a token reply from a completed sign-in.
func (m *Manager) Save(tr discovery.TokenResponse) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, err := m.persist(tr, "")
	if err == nil {
		m.cur = s
	}
	return err
}

// Current returns the stored sign-in without refreshing it.
func (m *Manager) Current() (*Set, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur == nil {
		s, err := m.load()
		if err != nil {
			return nil, err
		}
		m.cur = s
	}
	cp := *m.cur
	return &cp, nil
}

// Access returns a token valid for at least another minute, refreshing when needed.
func (m *Manager) Access(ctx context.Context) (string, error) {
	cur, err := m.Current()
	if err != nil {
		return "", err
	}
	if m.fresh(cur) {
		return cur.AccessToken, nil
	}
	return m.Refresh(ctx, cur.AccessToken)
}

// Refresh replaces stale; callers that lose the race, in this or another process, get the winner's token.
func (m *Manager) Refresh(ctx context.Context, stale string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur != nil && m.cur.AccessToken != stale && m.fresh(m.cur) {
		return m.cur.AccessToken, nil
	}
	unlock, err := fsutil.Lock(m.lock)
	if err != nil {
		return "", err
	}
	defer unlock()
	disk, err := m.load()
	if err != nil {
		m.cur = nil
		return "", err
	}
	if disk.AccessToken != stale && m.fresh(disk) {
		m.cur = disk
		return disk.AccessToken, nil
	}
	if disk.RefreshToken == "" {
		return "", ErrSignedOut
	}
	tr, err := oauth.Refresh(ctx, m.client, m.doc, disk.RefreshToken)
	if oauth.Is(err, "invalid_grant") {
		_ = m.store.Delete(Key)
		m.cur = nil
		return "", ErrSignedOut
	}
	if err != nil {
		return "", err
	}
	s, err := m.persist(tr, disk.RefreshToken)
	if err != nil {
		return "", err
	}
	m.cur = s
	return s.AccessToken, nil
}

// SignOut revokes the refresh token at Starlight, best effort, and forgets the sign-in locally.
func (m *Manager) SignOut(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, err := m.load(); err == nil && s.RefreshToken != "" && m.doc.RevocationEndpoint != "" {
		rctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		_ = oauth.Revoke(rctx, m.client, m.doc, s.RefreshToken, "refresh_token")
		cancel()
	}
	m.cur = nil
	return m.store.Delete(Key)
}
