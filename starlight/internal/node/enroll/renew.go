package enroll

import (
	"context"
	"crypto/x509"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/node/controlclient"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
)

// NeedsRenewal reports whether the certificate is due: Starlight's renew_after passed or a third of its life is left.
func NeedsRenewal(leaf *x509.Certificate, renewAfter int64, now time.Time) bool {
	if leaf == nil {
		return false
	}
	if renewAfter > 0 && now.After(time.Unix(renewAfter, 0)) {
		return true
	}
	life := leaf.NotAfter.Sub(leaf.NotBefore)
	return leaf.NotAfter.Sub(now) < life/3
}

// Renew rotates the node key, verifies the new certificate and swaps it in on disk and in memory.
func Renew(ctx context.Context, dir *state.Dir, c *controlclient.Client, id state.Identity, creds *state.Creds) error {
	_, ca, err := dir.WorkloadCA()
	if err != nil {
		return err
	}
	key, keyPEM, err := pki.NewKey()
	if err != nil {
		return err
	}
	csr, err := pki.CSR(key)
	if err != nil {
		return err
	}
	resp, err := c.Renew(ctx, wire.RenewRequest{CSR: string(csr)})
	if err != nil {
		return err
	}
	if err := verifyLeaf([]byte(resp.Certificate), ca, id.TrustDomain, id.NodeID, key); err != nil {
		return err
	}
	if err := dir.SaveCredentials([]byte(resp.Certificate), keyPEM); err != nil {
		return err
	}
	pair, err := state.ParsePair(append([]byte(resp.Certificate), keyPEM...))
	if err != nil {
		return err
	}
	creds.Set(pair)
	return nil
}
