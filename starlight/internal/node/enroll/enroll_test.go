package enroll_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/controlclient"
	"github.com/ashhart/TensorFold/starlight/internal/node/enroll"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
	"github.com/ashhart/TensorFold/starlight/internal/node/trust"
)

func dir(t *testing.T) *state.Dir {
	t.Helper()
	d, err := state.Open(filepath.Join(t.TempDir(), "state"))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestEnrolWithCAPinThenSyncAndRenew(t *testing.T) {
	ctl := nodetest.NewControl(t)
	d := dir(t)
	ctx := context.Background()
	id, resp, err := enroll.Run(ctx, d, enroll.Options{Server: ctl.URL(), Token: ctl.Token, Name: "mac-studio-01", CAPin: ctl.CAPin()})
	if err != nil {
		t.Fatal(err)
	}
	if id.NodeID != ctl.NodeID() || resp.Status != "active" || id.Trust != trust.CAPin || id.TrustDomain != "starlight.example" {
		t.Fatalf("identity %+v resp %+v", id, resp)
	}
	fi, _ := os.Stat(d.Root)
	if fi.Mode().Perm() != 0o700 {
		t.Fatalf("state dir mode %v", fi.Mode().Perm())
	}
	for _, f := range []string{"node.json", "node.pem", "workload-ca.pem", "permit-keys.json"} {
		fi, err := os.Stat(d.Path(f))
		if err != nil || fi.Mode().Perm() != 0o600 {
			t.Fatalf("%s: %v %v", f, err, fi)
		}
	}
	creds, err := state.LoadCreds(d)
	if err != nil {
		t.Fatal(err)
	}
	c, err := enroll.ControlClient(d, id, creds, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Sync(ctx, wire.SyncRequest{Generation: 0}); err != nil {
		t.Fatalf("sync with the node certificate: %v", err)
	}
	before := creds.Leaf().SerialNumber
	if err := enroll.Renew(ctx, d, c, id, creds); err != nil {
		t.Fatal(err)
	}
	if creds.Leaf().SerialNumber.Cmp(before) == 0 || ctl.Renewals() != 1 {
		t.Fatal("renewal did not swap the certificate")
	}
	reloaded, _ := state.LoadCreds(d)
	if reloaded.Leaf().SerialNumber.Cmp(creds.Leaf().SerialNumber) != 0 {
		t.Fatal("renewed certificate not persisted")
	}
	if _, err := c.Sync(ctx, wire.SyncRequest{Generation: 1}); err != nil {
		t.Fatalf("sync after renewal: %v", err)
	}
	if _, _, err := enroll.Run(ctx, d, enroll.Options{Server: ctl.URL(), Token: ctl.Token, CAPin: ctl.CAPin()}); err == nil {
		t.Fatal("re-enrolled without --force")
	}
}

func TestEnrolTrustFailures(t *testing.T) {
	ctl := nodetest.NewControl(t)
	ctx := context.Background()
	if _, _, err := enroll.Run(ctx, dir(t), enroll.Options{Server: ctl.URL(), Token: ctl.Token}); err == nil {
		t.Fatal("system roots trusted a private CA")
	}
	wrong := strings.Repeat("ab", 32)
	if _, _, err := enroll.Run(ctx, dir(t), enroll.Options{Server: ctl.URL(), Token: ctl.Token, CAPin: wrong}); err == nil {
		t.Fatal("accepted a mismatched CA pin")
	}
	if _, _, err := enroll.Run(ctx, dir(t), enroll.Options{Server: ctl.URL(), Token: "sle_wrong", CAPin: ctl.CAPin()}); !controlclient.Revoked(err) {
		t.Fatalf("bad token = %v", err)
	}
	if _, _, err := enroll.Run(ctx, dir(t), enroll.Options{Server: ctl.URL(), Token: ctl.Token, CABundle: ctl.CABundle()}); err != nil {
		t.Fatalf("ca-file trust: %v", err)
	}
	if _, _, err := enroll.Run(ctx, dir(t), enroll.Options{Server: "http://" + strings.TrimPrefix(ctl.URL(), "https://"), Token: ctl.Token, CAPin: ctl.CAPin()}); err == nil {
		t.Fatal("accepted a plain http server URL")
	}
}

func TestEnrolRejectsTamperedResponses(t *testing.T) {
	ctx := context.Background()
	other, keyDER, _ := pki.NewCA("starlight.example", time.Hour)
	other, _ = pki.LoadCA(other.CertPEM, keyDER)
	cases := map[string]func(*wire.EnrollResponse){
		"another node id": func(r *wire.EnrollResponse) { r.NodeID = ids.New() },
		"another CA": func(r *wire.EnrollResponse) {
			r.CA = string(other.CertPEM)
		},
		"another key": func(r *wire.EnrollResponse) {
			k, _, _ := pki.NewKey()
			cert, _, _ := other.Issue(pki.KindNode, r.NodeID, &k.PublicKey, time.Hour)
			r.Certificate, r.CA = string(cert), string(other.CertPEM)
		},
		"another trust domain": func(r *wire.EnrollResponse) { r.TrustDomain = "evil.example" },
		"invalid node id":      func(r *wire.EnrollResponse) { r.NodeID = "../../etc" },
	}
	for name, hook := range cases {
		ctl := nodetest.NewControl(t)
		ctl.EnrollHook = hook
		d := dir(t)
		if _, _, err := enroll.Run(ctx, d, enroll.Options{Server: ctl.URL(), Token: ctl.Token, CABundle: ctl.CABundle()}); err == nil {
			t.Errorf("%s: accepted", name)
		}
		if d.Enrolled() {
			t.Errorf("%s: wrote an identity for a rejected enrolment", name)
		}
	}
}

func TestNeedsRenewal(t *testing.T) {
	ca, keyDER, _ := pki.NewCA("starlight.example", 48*time.Hour)
	ca, _ = pki.LoadCA(ca.CertPEM, keyDER)
	k, _, _ := pki.NewKey()
	_, leaf, _ := ca.Issue(pki.KindNode, "n1", &k.PublicKey, 30*time.Hour)
	now := time.Now()
	if enroll.NeedsRenewal(leaf, 0, now) {
		t.Fatal("fresh certificate due")
	}
	if !enroll.NeedsRenewal(leaf, now.Add(-time.Second).Unix(), now) {
		t.Fatal("renew_after in the past ignored")
	}
	if !enroll.NeedsRenewal(leaf, 0, now.Add(25*time.Hour)) {
		t.Fatal("certificate with under a third of its life left not due")
	}
	if enroll.NeedsRenewal(nil, 0, now) {
		t.Fatal("nil certificate due")
	}
}

func TestControlClientRefusesNonControlPeers(t *testing.T) {
	ctl := nodetest.NewControl(t)
	d := dir(t)
	id, _, err := enroll.Run(context.Background(), d, enroll.Options{Server: ctl.URL(), Token: ctl.Token, CAPin: ctl.CAPin()})
	if err != nil {
		t.Fatal(err)
	}
	impostor := nodetest.NewControl(t)
	creds, _ := state.LoadCreds(d)
	id.ServerURL = impostor.URL()
	c, err := enroll.ControlClient(d, id, creds, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Sync(context.Background(), wire.SyncRequest{}); err == nil || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("talked to a control endpoint under another CA: %v", err)
	}
}
