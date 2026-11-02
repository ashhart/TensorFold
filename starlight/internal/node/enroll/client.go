package enroll

import (
	"fmt"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/node/controlclient"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
	"github.com/ashhart/TensorFold/starlight/internal/node/trust"
)

// TrustOf is the post-enrolment trust: the pinned workload CA and a control identity in the node's trust domain.
func TrustOf(dir *state.Dir, id state.Identity) (trust.Mode, error) {
	pool, _, err := dir.WorkloadCA()
	if err != nil {
		return trust.Mode{}, fmt.Errorf("workload CA: %w", err)
	}
	return trust.Mode{Kind: trust.Workload, Pool: pool, TrustDomain: id.TrustDomain}, nil
}

// ControlClient returns a client that presents the node certificate to Starlight.
func ControlClient(dir *state.Dir, id state.Identity, creds *state.Creds, timeout time.Duration) (*controlclient.Client, error) {
	mode, err := TrustOf(dir, id)
	if err != nil {
		return nil, err
	}
	httpc, base, err := mode.Client(id.ServerURL, creds.Get, nil, timeout)
	if err != nil {
		return nil, err
	}
	return &controlclient.Client{HTTP: httpc, Base: base}, nil
}
