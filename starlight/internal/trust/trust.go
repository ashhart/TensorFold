// Package trust holds Starlight's workload CA and permit keys, sealed in Postgres and shared by every instance.
package trust

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

const (
	caName       = "workload-ca"
	caValidity   = 10 * 365 * 24 * time.Hour
	leafValidity = 24 * time.Hour
)

// Bootstrap creates the CA and the first permit key; it is idempotent.
func Bootstrap(ctx context.Context, q db.Querier, keys *keyring.Keyring, trustDomain string) error {
	var exists bool
	if err := q.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM pki WHERE name = $1)", caName).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		ca, keyDER, err := pki.NewCA(trustDomain, caValidity)
		if err != nil {
			return err
		}
		sealed, err := keys.Seal(keyDER, "pki.key:"+caName)
		if err != nil {
			return err
		}
		if _, err := q.Exec(ctx, "INSERT INTO pki (name, cert_pem, key_sealed, not_after) VALUES ($1, $2, $3, $4)",
			caName, string(ca.CertPEM), sealed, ca.Cert.NotAfter); err != nil {
			return err
		}
	}
	var active int
	if err := q.QueryRow(ctx, "SELECT count(*) FROM permit_keys WHERE status = 'active'").Scan(&active); err != nil {
		return err
	}
	if active == 0 {
		return insertPermitKey(ctx, q, keys, "active")
	}
	return nil
}

// PublishWindow is how long nodes hold a new permit key before gateways sign with it; syncs reach nodes within 30 s.
const PublishWindow = 2 * time.Minute

// verifyFor keeps a replaced key verifiable while gateways and in-flight permits move to its successor.
const verifyFor = 48 * time.Hour

// RotatePermitKey publishes a new key to nodes now; gateways promote it and sign with it once PublishWindow passes.
func RotatePermitKey(ctx context.Context, q db.Querier, keys *keyring.Keyring) error {
	if _, err := q.Exec(ctx, "DELETE FROM permit_keys WHERE status = 'next'"); err != nil {
		return err
	}
	return insertPermitKey(ctx, q, keys, "next")
}

func insertPermitKey(ctx context.Context, q db.Querier, keys *keyring.Keyring, status string) error {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	kid := hex.EncodeToString(pub[:6])
	sealed, err := keys.Seal(priv, "permit_keys.private:"+kid)
	if err != nil {
		return err
	}
	_, err = q.Exec(ctx, "INSERT INTO permit_keys (kid, public_key, private_sealed, status) VALUES ($1, $2, $3, $4)", kid, []byte(pub), sealed, status)
	return err
}

// advance promotes a published key whose window has passed and retires keys replaced more than verifyFor ago.
func advance(ctx context.Context, pool *pgxpool.Pool) error {
	return db.Tx(ctx, pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, "UPDATE permit_keys SET status = 'retired', changed_at = now() WHERE status = 'verify' AND changed_at < $1",
			time.Now().Add(-verifyFor)); err != nil {
			return err
		}
		var kid string
		err := tx.QueryRow(ctx, "SELECT kid FROM permit_keys WHERE status = 'next' AND changed_at < $1 FOR UPDATE", time.Now().Add(-PublishWindow)).Scan(&kid)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil
		}
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, "UPDATE permit_keys SET status = 'verify', changed_at = now() WHERE status = 'active'"); err != nil {
			return err
		}
		_, err = tx.Exec(ctx, "UPDATE permit_keys SET status = 'active', changed_at = now() WHERE kid = $1", kid)
		return err
	})
}

// Trust is one instance's loaded view, refreshed in the background so rotations propagate.
type Trust struct {
	pool        *pgxpool.Pool
	keys        *keyring.Keyring
	instance    string
	mu          sync.RWMutex
	ca          *pki.CA
	signer      permit.Signer
	public      []wire.PermitKey
	gateway     *tls.Certificate
	control     *tls.Certificate
	gatewayExp  time.Time
	controlExp  time.Time
	TrustDomain string
}

// Load reads the CA and permit keys.
func Load(ctx context.Context, pool *pgxpool.Pool, keys *keyring.Keyring, instance string) (*Trust, error) {
	t := &Trust{pool: pool, keys: keys, instance: instance}
	if err := t.Refresh(ctx); err != nil {
		return nil, err
	}
	return t, nil
}

// Refresh reloads keys from Postgres, first moving any permit key whose window has passed.
func (t *Trust) Refresh(ctx context.Context) error {
	var certPEM string
	var sealed []byte
	if err := t.pool.QueryRow(ctx, "SELECT cert_pem, key_sealed FROM pki WHERE name = $1", caName).Scan(&certPEM, &sealed); err != nil {
		return errors.New("trust: no workload CA; run `starlight init`")
	}
	keyDER, err := t.keys.Open(sealed, "pki.key:"+caName)
	if err != nil {
		return err
	}
	ca, err := pki.LoadCA([]byte(certPEM), keyDER)
	if err != nil {
		return err
	}
	var due bool
	if err := t.pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM permit_keys WHERE status = 'next' AND changed_at < $1 OR status = 'verify' AND changed_at < $2)",
		time.Now().Add(-PublishWindow), time.Now().Add(-verifyFor)).Scan(&due); err != nil {
		return err
	}
	if due {
		if err := advance(ctx, t.pool); err != nil {
			return err
		}
	}
	rows, err := t.pool.Query(ctx, "SELECT kid, public_key, private_sealed, status FROM permit_keys WHERE status <> 'retired'")
	if err != nil {
		return err
	}
	defer rows.Close()
	var signer permit.Signer
	var public []wire.PermitKey
	for rows.Next() {
		var kid, status string
		var pub, sealedPriv []byte
		if err := rows.Scan(&kid, &pub, &sealedPriv, &status); err != nil {
			return err
		}
		public = append(public, wire.PermitKey{KID: kid, PublicKey: pub})
		if status == "active" {
			priv, err := t.keys.Open(sealedPriv, "permit_keys.private:"+kid)
			if err != nil {
				return err
			}
			signer = permit.Signer{KID: kid, Key: ed25519.PrivateKey(priv)}
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if signer.KID == "" {
		return errors.New("trust: no active permit key")
	}
	t.mu.Lock()
	t.ca, t.signer, t.public, t.TrustDomain = ca, signer, public, ca.TrustDomain
	t.mu.Unlock()
	return nil
}

// Run refreshes every minute until ctx ends.
func (t *Trust) Run(ctx context.Context) {
	tick := time.NewTicker(time.Minute)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			_ = t.Refresh(ctx)
		}
	}
}

// CA returns the workload CA.
func (t *Trust) CA() *pki.CA {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.ca
}

// Roots is a pool holding only the workload CA.
func (t *Trust) Roots() *x509.CertPool { return t.CA().Pool() }

// Signer returns the active permit signer.
func (t *Trust) Signer() (permit.Signer, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.signer, nil
}

// PublicKeys lists keys nodes should accept.
func (t *Trust) PublicKeys() []wire.PermitKey {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return append([]wire.PermitKey(nil), t.public...)
}

// Fingerprint is the hex SHA-256 of the CA certificate, printed in enrolment commands.
func (t *Trust) Fingerprint() string {
	sum := sha256.Sum256(t.CA().Cert.Raw)
	return hex.EncodeToString(sum[:])
}

// GatewayCert is this instance's client certificate for node calls, reissued before it expires.
func (t *Trust) GatewayCert() (*tls.Certificate, error) {
	return t.leaf(pki.KindGateway, &t.gateway, &t.gatewayExp)
}

// ControlCert is this instance's server certificate for the node listener.
func (t *Trust) ControlCert() (*tls.Certificate, error) {
	return t.leaf(pki.KindControl, &t.control, &t.controlExp)
}

func (t *Trust) leaf(kind pki.Kind, slot **tls.Certificate, exp *time.Time) (*tls.Certificate, error) {
	t.mu.RLock()
	c, e := *slot, *exp
	t.mu.RUnlock()
	if c != nil && time.Until(e) > leafValidity/2 {
		return c, nil
	}
	key, keyPEM, err := pki.NewKey()
	if err != nil {
		return nil, err
	}
	ca := t.CA()
	certPEM, cert, err := ca.Issue(kind, t.instance, &key.PublicKey, leafValidity)
	if err != nil {
		return nil, err
	}
	pair, err := tls.X509KeyPair(append(certPEM, ca.CertPEM...), keyPEM)
	if err != nil {
		return nil, err
	}
	t.mu.Lock()
	*slot, *exp = &pair, cert.NotAfter
	t.mu.Unlock()
	return &pair, nil
}
