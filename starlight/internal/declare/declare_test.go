package declare_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/declare"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

const example = "../../deploy/examples/fleet.yaml"

type rig struct {
	t       *testing.T
	ctx     context.Context
	pool    *pgxpool.Pool
	svc     *fleet.Service
	secrets map[string]string
}

func newRig(t *testing.T) *rig {
	pool := testutil.DB(t)
	r := &rig{t: t, ctx: context.Background(), pool: pool, svc: &fleet.Service{Pool: pool, Keys: keyring.ForTest()},
		secrets: map[string]string{"env:OPENAI_API_KEY": "sk-openai-test", "file:/etc/starlight/credentials/anthropic": "sk-ant-test"}}
	r.exec("INSERT INTO settings (installation_id, trust_domain) VALUES ($1, 'starlight-test')", ids.New())
	r.exec("INSERT INTO idp_connections (id, slug, name, protocol, preset) VALUES ($1, 'okta', 'Okta', 'oidc', 'okta')", ids.New())
	for _, email := range []string{"lead@example.com", "platform@example.com"} {
		if _, err := directory.CreateUser(r.ctx, pool, directory.NewUser{Email: email, Source: "scim"}); err != nil {
			t.Fatal(err)
		}
	}
	return r
}

func (r *rig) exec(sql string, args ...any) {
	r.t.Helper()
	if _, err := r.pool.Exec(r.ctx, sql, args...); err != nil {
		r.t.Fatalf("%s: %v", sql, err)
	}
}

func (r *rig) count(sql string, args ...any) int {
	r.t.Helper()
	var n int
	if err := r.pool.QueryRow(r.ctx, sql, args...).Scan(&n); err != nil {
		r.t.Fatalf("%s: %v", sql, err)
	}
	return n
}

func (r *rig) apply(f *declare.File, dry bool) (declare.Result, error) {
	return declare.Apply(r.ctx, r.svc, f, declare.Options{DryRun: dry, Secrets: func(ref string) (string, error) {
		if v, ok := r.secrets[ref]; ok {
			return v, nil
		}
		return declare.Reference(ref)
	}})
}

func (r *rig) example() *declare.File {
	r.t.Helper()
	f, err := declare.LoadFile(example)
	if err != nil {
		r.t.Fatal(err)
	}
	return f
}

func lines(res declare.Result) string {
	var out []string
	for _, c := range res.Changes {
		out = append(out, c.String())
	}
	return strings.Join(out, "\n")
}

func TestExampleAppliesOnceAndThenIsANoOp(t *testing.T) {
	r := newRig(t)
	f := r.example()
	dry, err := r.apply(f, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(dry.Changes) < 20 || r.count("SELECT count(*) FROM providers") != 0 || r.count("SELECT count(*) FROM audit_events") != 0 ||
		r.count("SELECT count(*) FROM groups") != 0 {
		t.Fatalf("dry run wrote, or found too little to do:\n%s", lines(dry))
	}
	applied, err := r.apply(f, false)
	if err != nil {
		t.Fatal(err)
	}
	if lines(applied) != lines(dry) {
		t.Fatalf("the dry run differs from the apply:\n%s\n---\n%s", lines(dry), lines(applied))
	}
	for _, want := range []string{"+ provider anthropic-main: anthropic at https://api.anthropic.com; credential from file:/etc/starlight/credentials/anthropic",
		"+ deployment coder-large-mac: team/coder-large from hf:example-org/coder-large@0123456789abcdef0123456789abcdef01234567",
		"+ route team/coder-large → anthropic-main/frontier-large", "+ global alias coder: → team/coder-large", "+ grant team/coder-large to every project",
		"+ group okta/research-engineers", "+ role member for service account ci-runner in research", "+ limit requests_per_minute per member of research: 60",
		"+ limit cost_per_month: $2500.00", "+ role auditor for group okta/starlight-auditors in the organisation", `~ settings: org_name "TensorFold Starlight" → "Example Corp"`} {
		if !strings.Contains(lines(applied), want) {
			t.Errorf("missing change %q in:\n%s", want, lines(applied))
		}
	}
	for sql, want := range map[string]int{
		"SELECT count(*) FROM providers WHERE credential IS NOT NULL":                                          2,
		"SELECT count(*) FROM deployments WHERE status = 'active'":                                             3,
		"SELECT count(*) FROM deployments WHERE spec->>'tensor_parallel' = 'glm' AND (spec->>'gpus')::int = 2": 1,
		"SELECT count(*) FROM model_grants WHERE project_id IS NULL":                                           2,
		"SELECT count(*) FROM role_bindings WHERE project_id IS NULL":                                          2,
		"SELECT count(*) FROM role_bindings WHERE project_id IS NOT NULL":                                      3,
		"SELECT count(*) FROM limits":                                                                          4,
		"SELECT count(*) FROM audit_events WHERE actor_type <> 'operator'":                                     0,
		"SELECT count(*) FROM settings WHERE org_name = 'Example Corp'":                                        1,
	} {
		if got := r.count(sql); got != want {
			t.Errorf("%s = %d, want %d", sql, got, want)
		}
	}
	err = r.svc.Do(r.ctx, audit.Event{ActorType: "operator", ActorID: "test"}, func(tx *fleet.Tx) error {
		pr, err := fleet.ProviderByName(r.ctx, tx.Q(), "openai-main")
		if err != nil {
			return err
		}
		if ok, err := tx.CredentialMatches(r.ctx, pr.ID, "sk-openai-test"); err != nil || !ok {
			return errors.Join(err, errors.New("the stored credential does not match its reference"))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	events := r.count("SELECT count(*) FROM audit_events")
	again, err := r.apply(f, false)
	if err != nil || len(again.Changes) != 0 || r.count("SELECT count(*) FROM audit_events") != events {
		t.Fatalf("second apply was not a no-op (%v):\n%s", err, lines(again))
	}
	var out strings.Builder
	if err := again.Write(&out); err != nil || !strings.Contains(out.String(), "No changes") {
		t.Fatalf("summary %q", out.String())
	}
}

func TestEditsShowAsReadableUpdates(t *testing.T) {
	r := newRig(t)
	f := r.example()
	if _, err := r.apply(f, false); err != nil {
		t.Fatal(err)
	}
	offline, retention := true, 730
	f.Settings.Offline, f.Settings.AuditRetentionDays = &offline, &retention
	f.Models[0].Deployments[0].Spec.Context = 65536
	f.Models[0].Deployments[0].Status = "paused"
	f.Projects[0].Limits[0].Value = "6000000"
	f.Projects[0].Aliases["frontier-coder"] = "team/embed"
	r.secrets["env:OPENAI_API_KEY"] = "sk-rotated"
	res, err := r.apply(f, false)
	if err != nil {
		t.Fatal(err)
	}
	got := lines(res)
	for _, want := range []string{"~ settings: offline false → true; audit_retention_days 400 → 730", "~ deployment coder-large-mac: spec.context 131072 → 65536 (new revision)",
		`~ deployment coder-large-mac: status "active" → "paused"`, "~ provider openai-main: credential replaced from env:OPENAI_API_KEY",
		`~ limit tokens_per_day in research: value "5,000,000" → "6,000,000"`, `~ alias in research frontier-coder: model "team/coder-large" → "team/embed"`} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if len(res.Changes) != 6 || strings.Contains(got, "sk-rotated") {
		t.Fatalf("unexpected changes, or a secret in the diff:\n%s", got)
	}
	if r.count("SELECT revision FROM deployments WHERE name = 'coder-large-mac'") != 2 {
		t.Fatal("the spec change did not start a new revision")
	}
}

func TestBadFilesChangeNothing(t *testing.T) {
	r := newRig(t)
	for text, want := range map[string]string{
		"version: 1\nmodles: []\n":                           "field modles not found",
		"database:\n  url: postgres://localhost/starlight\n": "server configuration",
		"providers: []\n":                                    "version: 1",
		"version: 1\nmodels:\n  - name: m\n    deployments:\n      - name: d\n        spec: {model: {kind: path, path: /m}, flags: [--fast]}\n": "unknown field",
		"version: 1\n---\nversion: 1\n": "exactly one YAML document",
	} {
		if _, err := declare.Load(strings.NewReader(text)); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("loading %q: %v", text, err)
		}
	}
	inline, _ := declare.Load(strings.NewReader("version: 1\nproviders:\n  - name: openai-main\n    kind: openai\n    credential: sk-live-123\n"))
	if _, err := declare.Apply(r.ctx, r.svc, inline, declare.Options{}); err == nil || !strings.Contains(err.Error(), "inline secrets are refused") ||
		strings.Contains(err.Error(), "sk-live-123") {
		t.Fatalf("inline credential: %v", err)
	}
	if _, err := declare.Reference("env:STARLIGHT_TEST_UNSET_VARIABLE"); err == nil {
		t.Fatal("an unset environment variable resolved")
	}
	if _, err := declare.Reference("file:relative/path"); err == nil {
		t.Fatal("a relative credential path resolved")
	}
	f := r.example()
	f.Projects[0].Members = append(f.Projects[0].Members, declare.Role{Role: "member", User: "nobody@example.com"})
	if _, err := r.apply(f, false); err == nil || !strings.Contains(err.Error(), "project research") || !strings.Contains(err.Error(), "nobody@example.com") {
		t.Fatalf("unknown member: %v", err)
	}
	if r.count("SELECT count(*) FROM providers") != 0 || r.count("SELECT count(*) FROM models") != 0 || r.count("SELECT count(*) FROM audit_events") != 0 {
		t.Fatal("a failed apply left partial changes")
	}
}
