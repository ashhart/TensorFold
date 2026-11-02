package fleetpages

import (
	"bytes"
	"net/http"
	"net/url"
	"slices"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

const day = "2006-01-02"

var usageGroups = []string{"project", "model", "user", "day"}

type usagePage struct {
	format
	From, To string
	Group    string
	Groups   []string
	Project  string
	Projects []fleet.Project
	Rows     []fleet.UsageRow
	Total    fleet.UsageRow
	Max      int64
	CSV      string
	Visible  bool
}

func (p *pages) usage() {
	p.c.Handle("GET /console/usage", "", p.usageReport)
	p.c.Handle("GET /console/usage.csv", "", p.usageCSV)
}

// usageQuery reads the report's filters from the query string and checks the caller may see the chosen projects.
func usageQuery(q *console.Request) (fleet.UsageFilter, usagePage, error) {
	v := q.R.URL.Query()
	page := usagePage{Group: v.Get("group"), Groups: usageGroups, Project: v.Get("project")}
	if !slices.Contains(usageGroups, page.Group) {
		page.Group = "project"
	}
	today := time.Now().UTC().Truncate(24 * time.Hour)
	from, err := dateParam(v, "from", today.AddDate(0, 0, -29))
	if err != nil {
		return fleet.UsageFilter{}, page, err
	}
	to, err := dateParam(v, "to", today)
	if err != nil {
		return fleet.UsageFilter{}, page, err
	}
	page.From, page.To = from.Format(day), to.Format(day)
	f := fleet.UsageFilter{From: from, To: to.AddDate(0, 0, 1), GroupBy: page.Group,
		Projects: visibleProjects(q, authz.UsageRead, authz.ProjectUsage)}
	page.Visible = f.Projects == nil || len(f.Projects) > 0
	if page.Project != "" {
		if !validID(page.Project) || f.Projects != nil && !slices.Contains(f.Projects, page.Project) {
			return f, page, forbidden()
		}
		f.Projects = []string{page.Project}
	}
	csv := url.Values{"from": {page.From}, "to": {page.To}, "group": {page.Group}}
	if page.Project != "" {
		csv.Set("project", page.Project)
	}
	page.CSV = "/console/usage.csv?" + csv.Encode()
	return f, page, nil
}

func dateParam(v url.Values, k string, def time.Time) (time.Time, error) {
	s := v.Get(k)
	if s == "" {
		return def, nil
	}
	t, err := time.Parse(day, s)
	if err != nil {
		return t, console.Invalid(k + " must be a date")
	}
	return t, nil
}

func (p *pages) usageReport(q *console.Request) error {
	f, page, err := usageQuery(q)
	if err != nil {
		return err
	}
	if page.Projects, err = fleet.Projects(q.Ctx(), p.c.Pool, visibleProjects(q, authz.UsageRead, authz.ProjectUsage)); err != nil {
		return err
	}
	if page.Visible {
		if page.Rows, page.Total, err = fleet.Usage(q.Ctx(), p.c.Pool, f); err != nil {
			return friendly(err)
		}
	}
	for _, r := range page.Rows {
		page.Max = max(page.Max, r.Tokens())
	}
	q.Render("fleet/usage", "Usage", page)
	return nil
}

func (p *pages) usageCSV(q *console.Request) error {
	f, page, err := usageQuery(q)
	if err != nil {
		return err
	}
	if !page.Visible {
		return forbidden()
	}
	f.Limit = fleet.ExportLimit
	rows, _, err := fleet.Usage(q.Ctx(), p.c.Pool, f)
	if err != nil {
		return friendly(err)
	}
	var buf bytes.Buffer
	if err := fleet.WriteCSV(&buf, page.Group, rows); err != nil {
		return err
	}
	h := q.W.Header()
	h.Set("Content-Type", "text/csv; charset=utf-8")
	h.Set("Content-Disposition", `attachment; filename="starlight-usage-`+page.Group+"-"+page.From+"-"+page.To+`.csv"`)
	h.Set("Cache-Control", "no-store")
	h.Set("X-Content-Type-Options", "nosniff")
	q.W.WriteHeader(http.StatusOK)
	_, err = q.W.Write(buf.Bytes())
	return err
}
