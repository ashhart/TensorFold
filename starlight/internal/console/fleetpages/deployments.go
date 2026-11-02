package fleetpages

import (
	"errors"
	"maps"
	"slices"
	"strconv"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

type deploymentPage struct {
	format
	New        bool
	Action     string
	Deployment fleet.Deployment
	Selector   string
	Replicas   []fleet.Replica
	CanWrite   bool
}

// artifactField feeds the artifact sub-form for the model weights or the drafter.
type artifactField struct {
	Prefix, Legend string
	Optional       bool
	A              wire.Artifact
}

// Artifact describes one artifact sub-form; the drafter is optional.
func (d deploymentPage) Artifact(prefix string) artifactField {
	if prefix == "drafter" {
		f := artifactField{Prefix: prefix, Legend: "Drafter", Optional: true}
		if d.Deployment.Spec.Drafter != nil {
			f.A = *d.Deployment.Spec.Drafter
		}
		return f
	}
	return artifactField{Prefix: prefix, Legend: "Model weights", A: d.Deployment.Spec.Model}
}

// Backends are the engine backends a spec may pin.
func (deploymentPage) Backends() []string { return []string{"metal", "cuda", "hip", "xpu"} }

// Efforts are the reasoning effort levels a spec may set.
func (deploymentPage) Efforts() []string { return []string{"low", "medium", "high", "xhigh"} }

// RankSchemes are the engine families Starlight can split over several GPUs and nodes.
func (deploymentPage) RankSchemes() []string { return slices.Sorted(maps.Keys(wire.RankSchemes)) }

func (d deploymentPage) title() string {
	if d.New {
		return "New deployment of " + d.Deployment.ModelName
	}
	return d.Deployment.Name
}

func (p *pages) deployments() {
	p.c.Handle("GET /console/models/{id}/deployments/new", authz.ModelsWrite, p.deploymentNew)
	p.c.Handle("GET /console/deployments/{name}", authz.ModelsRead, p.deploymentDetail)
	p.c.Handle("POST /console/deployments/{name}", authz.ModelsWrite, p.deploymentUpdate)
	p.c.Handle("POST /console/deployments/{name}/status", authz.ModelsWrite, p.deploymentStatus)
}

// fields reads form values and keeps the first error, so a rejected form is redrawn with everything that was typed.
type fields struct {
	q   *console.Request
	err error
}

func (f *fields) keep(err error) {
	if f.err == nil {
		f.err = err
	}
}

func (f *fields) int(k string, def int) int {
	n, err := f.q.Int(k, def)
	f.keep(err)
	return n
}

// artifact reads one artifact's fields under a prefix; an empty kind means none.
func (f *fields) artifact(prefix string, required bool) *wire.Artifact {
	q := f.q
	a := &wire.Artifact{Kind: q.Form(prefix + "_kind")}
	switch a.Kind {
	case "":
		if required {
			f.keep(console.Invalid("choose where the model weights come from"))
		}
		return nil
	case "hf":
		a.Repo, a.Revision = q.Form(prefix+"_repo"), q.Form(prefix+"_revision")
	case "path":
		a.Path = q.Form(prefix + "_path")
	case "url":
		a.URL, a.SHA256 = q.Form(prefix+"_url"), q.Form(prefix+"_sha256")
	default:
		f.keep(console.Invalid("artifact source must be hf, path or url"))
	}
	return a
}

// spec reads the typed engine spec; nodes build engine arguments only from these fields.
func (f *fields) spec() wire.Spec {
	q := f.q
	s := wire.Spec{Drafter: f.artifact("drafter", false), Backend: q.Form("backend"), KVCache: q.Form("kv_cache"),
		ReasoningEffort: q.Form("reasoning_effort"), Vision: q.Bool("vision"), NoDrafts: q.Bool("no_drafts"),
		Context: f.int("context", 0), Parallel: f.int("parallel", 0), ThinkingBudget: f.int("thinking_budget", 0),
		MaxTokens: f.int("max_tokens", 0), PromptCacheGiB: f.int("prompt_cache_gib", 0), PromptCache: q.Form("prompt_cache"), MemoryGiB: f.int("memory_gib", 0),
		GPUs: f.int("gpus", 0), Nodes: f.int("nodes", 0), TensorParallel: q.Form("tensor_parallel")}
	if m := f.artifact("model", true); m != nil {
		s.Model = *m
	}
	return s
}

func deploymentInput(q *console.Request, name string) (fleet.DeploymentInput, error) {
	f := &fields{q: q}
	in := fleet.DeploymentInput{Name: name, Spec: f.spec(), Pool: q.Form("pool"), Mode: q.Form("mode"), MinReplicas: f.int("min_replicas", 1),
		MaxReplicas: f.int("max_replicas", 0), IdleSeconds: f.int("idle_seconds", 900)}
	sel, err := parseLabels(q.Lines("selector"))
	f.keep(err)
	in.Selector = sel
	return in, f.err
}

// redraw shows the form again with the user's input when the error is one they can correct.
func redraw(q *console.Request, page deploymentPage, in fleet.DeploymentInput, err error) error {
	var inv console.Invalid
	if !errors.As(err, &inv) {
		return err
	}
	d := &page.Deployment
	d.Name, d.Spec, d.Pool, d.Mode = in.Name, in.Spec, in.Pool, in.Mode
	d.MinReplicas, d.MaxReplicas, d.IdleSeconds = in.MinReplicas, in.MaxReplicas, in.IdleSeconds
	page.Selector = q.R.PostFormValue("selector")
	return &console.Rerender{Page: "fleet/deployment", Title: page.title(), Data: page, Msg: string(inv)}
}

func (p *pages) deploymentNew(q *console.Request) error {
	m, err := fleet.ModelByID(q.Ctx(), p.c.Pool, q.R.PathValue("id"))
	if err != nil {
		return friendly(err)
	}
	d := fleet.Deployment{ModelID: m.ID, ModelName: m.Name, Pool: "default", Mode: "warm", MinReplicas: 1, MaxReplicas: 1, IdleSeconds: 900,
		Spec: wire.Spec{Model: wire.Artifact{Kind: "hf"}}}
	page := deploymentPage{New: true, Action: "/console/models/" + m.ID + "/deployments", Deployment: d, CanWrite: true}
	q.Render("fleet/deployment", page.title(), page)
	return nil
}

func (p *pages) deploymentCreate(q *console.Request, m fleet.Model) error {
	in, err := deploymentInput(q, q.Form("name"))
	in.ModelID = m.ID
	if err == nil {
		err = p.do(q, func(tx *fleet.Tx) error { _, err := tx.CreateDeployment(q.Ctx(), in); return err })
	}
	if err != nil {
		page := deploymentPage{New: true, Action: "/console/models/" + m.ID + "/deployments", CanWrite: true,
			Deployment: fleet.Deployment{ModelID: m.ID, ModelName: m.Name}}
		return redraw(q, page, in, err)
	}
	return q.Redirect("/console/deployments/"+in.Name, "Deployment created; the scheduler places it within seconds.")
}

func (p *pages) deploymentDetail(q *console.Request) error {
	d, err := fleet.DeploymentByName(q.Ctx(), p.c.Pool, q.R.PathValue("name"))
	if err != nil {
		return friendly(err)
	}
	reps, err := fleet.Replicas(q.Ctx(), p.c.Pool, d.ID, "")
	if err != nil {
		return err
	}
	page := deploymentPage{Action: "/console/deployments/" + d.Name, Deployment: d, Selector: joinPairs(d.Selector, "="), Replicas: reps,
		CanWrite: q.Can(authz.ModelsWrite, "")}
	q.Render("fleet/deployment", page.title(), page)
	return nil
}

func (p *pages) deploymentUpdate(q *console.Request) error {
	d, err := fleet.DeploymentByName(q.Ctx(), p.c.Pool, q.R.PathValue("name"))
	if err != nil {
		return friendly(err)
	}
	in, err := deploymentInput(q, d.Name)
	in.ModelID = d.ModelID
	var after fleet.Deployment
	if err == nil {
		err = p.do(q, func(tx *fleet.Tx) (err error) { after, err = tx.UpdateDeployment(q.Ctx(), d.ID, in); return err })
	}
	if err != nil {
		reps, rerr := fleet.Replicas(q.Ctx(), p.c.Pool, d.ID, "")
		if rerr != nil {
			return rerr
		}
		return redraw(q, deploymentPage{Action: "/console/deployments/" + d.Name, Deployment: d, Replicas: reps, CanWrite: true}, in, err)
	}
	notice := "Deployment saved."
	if after.Revision != d.Revision {
		notice = "Engine settings changed: revision " + strconv.Itoa(after.Revision) + " rolls out surge-first."
	}
	return q.Redirect("/console/deployments/"+d.Name, notice)
}

func (p *pages) deploymentStatus(q *console.Request) error {
	d, err := fleet.DeploymentByName(q.Ctx(), p.c.Pool, q.R.PathValue("name"))
	if err != nil {
		return friendly(err)
	}
	status := q.Form("status")
	if status == "retired" && !q.Bool("confirm") {
		return console.Invalid("Tick the confirmation box to retire " + d.Name + ".")
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.SetDeploymentStatus(q.Ctx(), d.ID, status) }); err != nil {
		return err
	}
	return q.Redirect("/console/deployments/"+d.Name, "Deployment "+status+".")
}
