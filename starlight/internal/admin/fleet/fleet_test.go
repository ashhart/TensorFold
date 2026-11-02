package fleet_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

type rig struct {
	t     *testing.T
	ctx   context.Context
	pool  *pgxpool.Pool
	svc   *fleet.Service
	actor audit.Event
}

func newRig(t *testing.T) *rig {
	pool := testutil.DB(t)
	return &rig{t: t, ctx: context.Background(), pool: pool, svc: &fleet.Service{Pool: pool, Keys: keyring.ForTest()},
		actor: audit.Event{ActorType: "user", ActorID: ids.New(), ActorLabel: "admin@example.com"}}
}

func (r *rig) do(f func(*fleet.Tx) error) error {
	return r.svc.Do(r.ctx, r.actor, f)
}

func (r *rig) must(f func(*fleet.Tx) error) {
	r.t.Helper()
	if err := r.do(f); err != nil {
		r.t.Fatal(err)
	}
}

func (r *rig) audited(action string) int {
	var n int
	_ = r.pool.QueryRow(r.ctx, "SELECT count(*) FROM audit_events WHERE action = $1", action).Scan(&n)
	return n
}

func invalid(err error) bool {
	var inv fleet.Invalid
	return errors.As(err, &inv)
}

func TestProjectsMembersGrantsAliases(t *testing.T) {
	r := newRig(t)
	var p fleet.Project
	r.must(func(tx *fleet.Tx) (err error) {
		p, err = tx.CreateProject(r.ctx, fleet.ProjectInput{Slug: "eng", Name: "Engineering"})
		return err
	})
	if p.Classification != "internal" || p.PlanApproval != "required" || p.SecretsGuard != "local" || r.audited("project.created") != 1 {
		t.Fatalf("project defaults %+v", p)
	}
	if err := r.do(func(tx *fleet.Tx) error {
		_, err := tx.CreateProject(r.ctx, fleet.ProjectInput{Slug: "eng", Name: "Again"})
		return err
	}); !invalid(err) {
		t.Fatalf("duplicate slug: %v", err)
	}
	if err := r.do(func(tx *fleet.Tx) error {
		_, err := tx.CreateProject(r.ctx, fleet.ProjectInput{Slug: "Bad Slug", Name: "x"})
		return err
	}); !invalid(err) {
		t.Fatalf("bad slug: %v", err)
	}
	r.must(func(tx *fleet.Tx) error {
		_, err := tx.UpdateProject(r.ctx, p.ID, fleet.ProjectInput{Slug: "eng", Name: "Engineering", Classification: "confidential", PlanApproval: "auto"})
		return err
	})
	u, _ := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "kim@example.com", Source: "jit"})
	r.must(func(tx *fleet.Tx) error {
		if _, err := tx.Bind(r.ctx, "project_admin", "user", u.ID, p.ID); err != nil {
			return err
		}
		ok, err := tx.Bind(r.ctx, "project_admin", "user", u.ID, p.ID)
		if ok {
			t.Error("duplicate binding reported as new")
		}
		return err
	})
	if err := r.do(func(tx *fleet.Tx) error { _, err := tx.Bind(r.ctx, "owner", "user", u.ID, p.ID); return err }); !invalid(err) {
		t.Fatalf("org role in a project: %v", err)
	}
	bs, err := fleet.Bindings(r.ctx, r.pool, p.ID)
	if err != nil || len(bs) != 1 || bs[0].Label != "kim@example.com" {
		t.Fatalf("bindings %+v %v", bs, err)
	}
	var m fleet.Model
	r.must(func(tx *fleet.Tx) (err error) {
		m, err = tx.CreateModel(r.ctx, fleet.ModelInput{Name: "qwen3.8-27b", Capabilities: fleet.Capabilities{ContextWindow: 131072, Tools: true}})
		return err
	})
	five := 5000
	r.must(func(tx *fleet.Tx) error {
		if err := tx.SetGrant(r.ctx, m.ID, "", nil, nil); err != nil {
			return err
		}
		if err := tx.SetGrant(r.ctx, m.ID, "", nil, &five); err != nil {
			return err
		}
		if err := tx.SetGrant(r.ctx, m.ID, p.ID, &five, nil); err != nil {
			return err
		}
		return tx.SetAlias(r.ctx, "claude-sonnet-4-5", "", m.ID)
	})
	gs, _ := fleet.Grants(r.ctx, r.pool, "", m.ID)
	if len(gs) != 2 || gs[0].ProjectID != "" || gs[0].MaxOutputTokens == nil || *gs[0].MaxOutputTokens != 5000 {
		t.Fatalf("grants %+v", gs)
	}
	if err := r.do(func(tx *fleet.Tx) error { return tx.SetAlias(r.ctx, "bad alias", "", m.ID) }); !invalid(err) {
		t.Fatalf("bad alias: %v", err)
	}
	al, _ := fleet.Aliases(r.ctx, r.pool, "global")
	if len(al) != 1 || al[0].ModelName != "qwen3.8-27b" {
		t.Fatalf("aliases %+v", al)
	}
	r.must(func(tx *fleet.Tx) error { return tx.RemoveAlias(r.ctx, "claude-sonnet-4-5", "") })
	if r.audited("model.granted") != 3 || r.audited("model.alias_removed") != 1 {
		t.Fatal("grant and alias changes were not audited")
	}
}

func TestProvidersSealCredentialsAndRoutesDefaultProtocols(t *testing.T) {
	r := newRig(t)
	var p fleet.Provider
	r.must(func(tx *fleet.Tx) (err error) {
		p, err = tx.CreateProvider(r.ctx, fleet.ProviderInput{Name: "anthropic", Kind: "anthropic", Credential: "sk-ant-test-123", Enabled: true,
			Headers: map[string]string{"anthropic-beta": "x"}})
		return err
	})
	if p.BaseURL != "https://api.anthropic.com" || !p.HasCredential || p.MaxClass != "internal" {
		t.Fatalf("provider %+v", p)
	}
	var raw []byte
	_ = r.pool.QueryRow(r.ctx, "SELECT credential FROM providers WHERE id = $1", p.ID).Scan(&raw)
	if strings.Contains(string(raw), "sk-ant") {
		t.Fatal("credential stored in plain text")
	}
	r.must(func(tx *fleet.Tx) error {
		ok, err := tx.CredentialMatches(r.ctx, p.ID, "sk-ant-test-123")
		if !ok {
			t.Error("sealed credential does not open")
		}
		return err
	})
	in := fleet.ProviderInput{Name: "anthropic", Kind: "anthropic", BaseURL: "https://api.anthropic.com", Enabled: true}
	r.must(func(tx *fleet.Tx) error { _, err := tx.UpdateProvider(r.ctx, p.ID, in); return err })
	got, _ := fleet.ProviderByName(r.ctx, r.pool, "anthropic")
	if !got.HasCredential {
		t.Fatal("an empty credential field erased the stored credential")
	}
	for _, bad := range []fleet.ProviderInput{{Name: "x1", Kind: "openai_compatible", BaseURL: "http://10.0.0.5:8000/v1"},
		{Name: "x2", Kind: "openai", Headers: map[string]string{"Authorization": "Bearer y"}}, {Name: "x3", Kind: "azure_openai"}} {
		if err := r.do(func(tx *fleet.Tx) error { _, err := tx.CreateProvider(r.ctx, bad); return err }); !invalid(err) {
			t.Errorf("accepted %+v: %v", bad, err)
		}
	}
	r.must(func(tx *fleet.Tx) error {
		_, err := tx.CreateProvider(r.ctx, fleet.ProviderInput{Name: "vllm", Kind: "openai_compatible", BaseURL: "http://10.0.0.5:8000/v1", Internal: true})
		return err
	})
	var m fleet.Model
	r.must(func(tx *fleet.Tx) (err error) {
		m, err = tx.CreateModel(r.ctx, fleet.ModelInput{Name: "frontier", RoutePolicy: "cloud_only"})
		if err != nil {
			return err
		}
		return tx.SetRoute(r.ctx, m.ID, fleet.RouteInput{ProviderID: p.ID, UpstreamModel: "claude-model", Enabled: true})
	})
	routes, _ := fleet.Routes(r.ctx, r.pool, m.ID)
	if len(routes) != 1 || len(routes[0].Protocols) != 3 || routes[0].MaxConcurrency != 64 {
		t.Fatalf("routes %+v", routes)
	}
	if err := r.do(func(tx *fleet.Tx) error {
		return tx.SetRoute(r.ctx, m.ID, fleet.RouteInput{ProviderID: p.ID, UpstreamModel: "m", Protocols: []string{"smtp"}})
	}); !invalid(err) {
		t.Fatalf("unknown protocol: %v", err)
	}
}

func TestDeploymentRevisionsAndValidation(t *testing.T) {
	r := newRig(t)
	var m fleet.Model
	r.must(func(tx *fleet.Tx) (err error) {
		m, err = tx.CreateModel(r.ctx, fleet.ModelInput{Name: "qwen"})
		return err
	})
	spec := wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/models/qwen"}, Parallel: 4}
	in := fleet.DeploymentInput{Name: "qwen-mac", ModelID: m.ID, Spec: spec, MinReplicas: 1}
	var d fleet.Deployment
	r.must(func(tx *fleet.Tx) (err error) { d, err = tx.CreateDeployment(r.ctx, in); return err })
	if d.Revision != 1 || d.Pool != "default" || d.MaxReplicas != 1 || d.IdleSeconds != 900 {
		t.Fatalf("deployment %+v", d)
	}
	r.must(func(tx *fleet.Tx) (err error) { d, err = tx.UpdateDeployment(r.ctx, d.ID, in); return err })
	if d.Revision != 1 {
		t.Fatalf("an unchanged spec bumped the revision to %d", d.Revision)
	}
	in.Spec.Context = 65536
	in.MaxReplicas = 3
	r.must(func(tx *fleet.Tx) (err error) { d, err = tx.UpdateDeployment(r.ctx, d.ID, in); return err })
	if d.Revision != 2 || d.MaxReplicas != 3 {
		t.Fatalf("spec change: %+v", d)
	}
	bad := in
	bad.Name = "qwen-hf"
	bad.Spec.Model = wire.Artifact{Kind: "hf", Repo: "TensorFold/Qwen", Revision: "main"}
	if err := r.do(func(tx *fleet.Tx) error { _, err := tx.CreateDeployment(r.ctx, bad); return err }); !invalid(err) {
		t.Fatalf("unpinned artifact accepted: %v", err)
	}
	bad.Spec.Model = wire.Artifact{Kind: "url", URL: "http://mirror.example/q.tar", SHA256: strings.Repeat("a", 64)}
	if err := r.do(func(tx *fleet.Tx) error { _, err := tx.CreateDeployment(r.ctx, bad); return err }); !invalid(err) {
		t.Fatalf("plain http artifact accepted: %v", err)
	}
	r.must(func(tx *fleet.Tx) error { return tx.SetDeploymentStatus(r.ctx, d.ID, "paused") })
	if r.audited("deployment.paused") != 1 || r.audited("deployment.updated") != 2 {
		t.Fatal("deployment changes were not audited")
	}
}

func TestUnbindingTheLastOwnerIsRefused(t *testing.T) {
	r := newRig(t)
	u, err := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "only-owner@example.com", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	binding := ids.New()
	if _, err := r.pool.Exec(r.ctx, "INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, 'owner', 'user', $2)", binding, u.ID); err != nil {
		t.Fatal(err)
	}
	if err := r.do(func(tx *fleet.Tx) error { return tx.Unbind(r.ctx, binding, "") }); err == nil {
		t.Fatal("the organisation's only owner binding was removed")
	}
	var n int
	_ = r.pool.QueryRow(r.ctx, "SELECT count(*) FROM role_bindings WHERE id = $1", binding).Scan(&n)
	if n != 1 {
		t.Fatal("the refused unbind still deleted the binding")
	}
}
