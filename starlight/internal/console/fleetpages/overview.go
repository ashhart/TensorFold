package fleetpages

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

// format gives templates the number formats the live script also uses.
type format struct{}

// Rate formats tokens per second.
func (format) Rate(v float64) string { return fleet.Rate(v) }

// Count formats a whole number with separators.
func (format) Count(v any) string {
	switch n := v.(type) {
	case int:
		return fleet.Count(int64(n))
	case int64:
		return fleet.Count(n)
	case *int64:
		if n != nil {
			return fleet.Count(*n)
		}
	}
	return ""
}

// Cap formats an optional token cap.
func (format) Cap(v *int) string {
	if v == nil {
		return "none"
	}
	return fleet.Count(int64(*v))
}

// Percent formats a share between 0 and 1.
func (format) Percent(v float64) string { return fleet.Percent(v) }

// Dollars formats micro-dollars.
func (format) Dollars(v int64) string { return fleet.Dollars(v) }

// Money formats micro-dollars for a form field, empty when zero.
func (format) Money(v int64) string {
	if v == 0 {
		return ""
	}
	return fleet.Dollars(v)
}

// Bytes formats a byte count in binary units.
func (format) Bytes(v uint64) string {
	switch {
	case v == 0:
		return "—"
	case v >= 1<<40:
		return strconv.FormatFloat(float64(v)/float64(1<<40), 'f', 1, 64) + " TiB"
	case v >= 1<<30:
		return strconv.FormatUint(v>>30, 10) + " GiB"
	}
	return strconv.FormatUint(v>>20, 10) + " MiB"
}

// Date formats an optional date for a date field.
func (format) Date(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.UTC().Format(day)
}

// Age formats seconds as a short duration.
func (format) Age(s int) string {
	switch {
	case s < 60:
		return strconv.Itoa(s) + "s"
	case s < 3600:
		return strconv.Itoa(s/60) + "m " + strconv.Itoa(s%60) + "s"
	}
	return strconv.Itoa(s/3600) + "h " + strconv.Itoa(s%3600/60) + "m"
}

type overviewPage struct {
	format
	Snap       fleet.Snapshot
	CanUsage   bool
	CanNodes   bool
	CanModels  bool
	CanControl bool
	PublicURL  string
	Next       string
}

// Spark draws the small throughput line in a KPI card.
func (o overviewPage) Spark(series string) string { return fleet.Spark(o.values(series), 120, 32) }

// SparkArea fills under the small line.
func (o overviewPage) SparkArea(series string) string { return fleet.Area(o.values(series), 120, 32) }

// Chart draws a 60-minute chart line.
func (o overviewPage) Chart(series string) string { return fleet.Spark(o.values(series), 600, 140) }

// ChartArea fills under a chart line.
func (o overviewPage) ChartArea(series string) string { return fleet.Area(o.values(series), 600, 140) }

func (o overviewPage) values(series string) []float64 {
	if series == "requests" {
		return o.Snap.Requests
	}
	return o.Snap.Throughput
}

// InFlightTotal is local plus cloud requests in flight.
func (o overviewPage) InFlightTotal() int {
	if o.Snap.Tiles == nil {
		return 0
	}
	return o.Snap.Tiles.InFlightLocal + o.Snap.Tiles.InFlightCloud
}

func scope(q *console.Request) fleet.Scope {
	org := q.Grants.AnyOrg()
	s := fleet.Scope{Org: org, UserID: q.User.ID, Usage: org && q.Can(authz.UsageRead, ""), Nodes: org && q.Can(authz.NodesRead, ""),
		Models: org && q.Can(authz.ModelsRead, "")}
	if !org {
		s.ProjectIDs = q.Grants.ProjectIDs(authz.Inference)
		if s.ProjectIDs == nil {
			s.ProjectIDs = []string{}
		}
	}
	return s
}

func (p *pages) overview() {
	p.c.Handle("GET /console", "", p.dashboard)
	p.c.Handle("GET /console/{$}", "", p.dashboard)
	p.c.Handle("GET /console/live.json", "", p.liveJSON)
}

func (p *pages) dashboard(q *console.Request) error {
	s := scope(q)
	snap, err := fleet.Load(q.Ctx(), p.c.Pool, s)
	if err != nil {
		return err
	}
	q.Render("fleet/overview", "Overview", overviewPage{Snap: snap, CanUsage: s.Usage, CanNodes: s.Nodes, CanModels: s.Models,
		CanControl: q.Can(authz.NodesWrite, ""), PublicURL: p.c.PublicURL, Next: "/console"})
	return nil
}

// liveJSON is the dashboard's polling feed, filtered by the same permissions as the page.
func (p *pages) liveJSON(q *console.Request) error {
	snap, err := fleet.Load(q.Ctx(), p.c.Pool, scope(q))
	if err != nil {
		return err
	}
	h := q.W.Header()
	h.Set("Content-Type", "application/json")
	h.Set("Cache-Control", "no-store")
	h.Set("X-Content-Type-Options", "nosniff")
	q.W.WriteHeader(http.StatusOK)
	return json.NewEncoder(q.W).Encode(snap)
}
