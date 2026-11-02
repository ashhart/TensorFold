package fleet_test

import (
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func TestSharingAnIsolatedModelRestartsItsPromptCaches(t *testing.T) {
	r := newRig(t)
	model, shared, node, first := r.fleetFixture()
	off := ids.New()
	r.exec(`INSERT INTO deployments (id, model_id, name, spec) VALUES ($1, $2, 'qwen-nocache', '{"model":{"kind":"path","path":"/m"},"prompt_cache":"off"}')`,
		off, model)
	replica := func(deployment string) string {
		id := ids.New()
		r.exec("INSERT INTO replicas (id, deployment_id, deployment_revision, node_id, state, slots, epoch) VALUES ($1, $2, 1, $3, 'ready', 4, 1)",
			id, deployment, node)
		return id
	}
	uncached := replica(off)
	desired := func(id string) string {
		var d string
		_ = r.pool.QueryRow(r.ctx, "SELECT desired FROM replicas WHERE id = $1", id).Scan(&d)
		return d
	}
	var vault, lab fleet.Project
	r.must(func(tx *fleet.Tx) (err error) {
		if vault, err = tx.CreateProject(r.ctx, fleet.ProjectInput{Slug: "vault", Name: "Vault", IsolateCache: true}); err != nil {
			return err
		}
		lab, err = tx.CreateProject(r.ctx, fleet.ProjectInput{Slug: "lab", Name: "Lab"})
		return err
	})
	five := 5000
	r.must(func(tx *fleet.Tx) error { return tx.SetGrant(r.ctx, model, vault.ID, nil, nil) })
	r.must(func(tx *fleet.Tx) error { return tx.SetGrant(r.ctx, model, vault.ID, nil, &five) })
	if desired(first) != "running" {
		t.Fatal("granting a model to the one isolated project using it restarted its replicas")
	}
	grant := func(project string) {
		r.must(func(tx *fleet.Tx) error { return tx.SetGrant(r.ctx, model, project, nil, nil) })
	}
	remove := func(project string) {
		gs, err := fleet.Grants(r.ctx, r.pool, "", model)
		for _, g := range gs {
			if g.ProjectID == project {
				r.must(func(tx *fleet.Tx) error { return tx.RemoveGrant(r.ctx, g.ID) })
				return
			}
		}
		t.Fatalf("no grant of the model to %q: %+v %v", project, gs, err)
	}
	grant(lab.ID)
	if desired(first) != "stopped" || desired(uncached) != "running" {
		t.Fatalf("another project granted the model: shared replica %s, cache-off replica %s", desired(first), desired(uncached))
	}
	remove(lab.ID)
	second := replica(shared)
	grant("")
	if desired(second) != "stopped" {
		t.Fatal("granting the model to every project kept the isolated project's prompt cache")
	}
	remove("")
	third := replica(shared)
	remove(vault.ID)
	if desired(third) != "stopped" || desired(uncached) != "running" {
		t.Fatalf("the isolated project lost the model: shared replica %s, cache-off replica %s", desired(third), desired(uncached))
	}
}
