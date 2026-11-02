// Package fleetpages serves the console's live overview and its project, model, node, provider and budget pages.
package fleetpages

import (
	"embed"
	"errors"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

//go:embed templates
var templateFS embed.FS

// Templates holds this package's pages under templates/fleet/.
var Templates = templateFS

// PlanDecider applies approvals with the plans service's own rules; the plans service implements it.
type PlanDecider interface {
	Approve(r *http.Request, approver *authn.Caller, id, planSHA256 string) error
	Reject(r *http.Request, approver *authn.Caller, id string) error
}

// Option configures Register.
type Option func(*pages)

// WithPlans lets approvers decide plans from the console; without it plans are shown read-only.
func WithPlans(d PlanDecider) Option { return func(p *pages) { p.plans = d } }

type pages struct {
	c     *console.Console
	svc   *fleet.Service
	plans PlanDecider
}

// Register mounts every fleet page on the console.
func Register(c *console.Console, opts ...Option) {
	p := &pages{c: c, svc: &fleet.Service{Pool: c.Pool, Keys: c.Keys, Flush: c.Flush}}
	for _, o := range opts {
		o(p)
	}
	p.overview()
	p.projects()
	p.models()
	p.deployments()
	p.nodes()
	p.providers()
	p.limits()
	p.usage()
	p.planPages()
	p.subscriptions()
}

// do runs a fleet change attributed to the signed-in user.
func (p *pages) do(q *console.Request, f func(*fleet.Tx) error) error {
	return friendly(p.svc.Do(q.Ctx(), q.Audit("", "", "", nil), f))
}

// friendly turns service errors into what the console shows.
func friendly(err error) error {
	var inv fleet.Invalid
	switch {
	case err == nil:
		return nil
	case errors.As(err, &inv):
		return console.Invalid(string(inv))
	case errors.Is(err, fleet.ErrNotFound), errors.Is(err, keys.ErrNotFound):
		return httpx.Errorf(http.StatusNotFound, "not_found", "That record does not exist.")
	}
	return err
}

func forbidden() error {
	return httpx.Errorf(http.StatusForbidden, "forbidden", "You do not have permission to do that.")
}

// caller is the signed-in user as the plans service and other API-level code expects.
func caller(q *console.Request) *authn.Caller {
	return &authn.Caller{P: q.Principal(), Grants: q.Grants}
}

// optInt reads an optional positive integer field.
func optInt(q *console.Request, k string) (*int, error) {
	v := q.Form(k)
	if v == "" {
		return nil, nil
	}
	n, err := q.Int(k, 0)
	if err != nil || n <= 0 {
		return nil, console.Invalid(k + " must be a positive whole number or empty")
	}
	return &n, nil
}

// int64Field reads a non-negative integer, or a dollar amount for cost kinds.
func int64Field(q *console.Request, k string, dollars bool) (int64, error) {
	if dollars {
		v, err := fleet.ParseDollars(k, q.Form(k))
		return v, friendly(err)
	}
	n, err := q.Int(k, 0)
	if err != nil || n < 0 {
		return 0, console.Invalid(k + " must be a non-negative whole number")
	}
	return int64(n), nil
}

// parseLabels reads key=value lines.
func parseLabels(lines []string) (map[string]string, error) {
	out := map[string]string{}
	for _, l := range lines {
		k, v, ok := strings.Cut(l, "=")
		if !ok || strings.TrimSpace(k) == "" {
			return nil, console.Invalid("labels are key=value, one per line")
		}
		out[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	return out, nil
}

// joinPairs writes a map back as sorted key, separator, value lines for an edit form.
func joinPairs(m map[string]string, sep string) string {
	var out []string
	for k, v := range m {
		out = append(out, k+sep+v)
	}
	sort.Strings(out)
	return strings.Join(out, "\n")
}

// localTime reads an HTML datetime-local field as UTC.
func localTime(q *console.Request, k string) (*time.Time, error) {
	v := q.Form(k)
	if v == "" {
		return nil, nil
	}
	t, err := time.Parse("2006-01-02T15:04", v)
	if err != nil {
		return nil, console.Invalid(k + " must be a date and time")
	}
	return &t, nil
}

// visibleProjects lists projects where the user holds projectPerm, or nil when an organisation role covers them all.
func visibleProjects(q *console.Request, orgPerm, projectPerm authz.Perm) []string {
	if q.Can(orgPerm, "") || q.Grants.EveryProject(projectPerm) {
		return nil
	}
	ids := q.Grants.ProjectIDs(projectPerm)
	if ids == nil {
		ids = []string{}
	}
	return ids
}

func validID(s string) bool { return ids.Valid(s) }

// secret is a value shown exactly once after it is created.
type secret struct {
	Label   string
	Value   string
	Command string
	Back    string
}
