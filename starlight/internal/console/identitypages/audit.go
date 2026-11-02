package identitypages

import (
	"encoding/hex"
	"encoding/json"
	"net/url"
	"strconv"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/admin/settings"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

type auditFilters struct {
	Action, Actor, Target, From, To string
}

type verifyResult struct {
	OK      bool
	Message string
}

type auditPage struct {
	Rows    []audit.Record
	Filters auditFilters
	Next    string
	Verify  *verifyResult
}

type eventPage struct {
	Record  audit.Record
	Details string
	Hash    string
	Prev    string
}

func day(s string, end bool) *time.Time {
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return nil
	}
	if end {
		t = t.Add(24 * time.Hour)
	}
	return &t
}

func (p pages) auditLog() {
	c := p.c
	c.Handle("GET /console/audit", authz.AuditRead, func(q *console.Request) error { return p.renderAudit(q, nil) })
	c.Handle("GET /console/audit/{seq}", authz.AuditRead, func(q *console.Request) error {
		seq, err := strconv.ParseInt(q.R.PathValue("seq"), 10, 64)
		if err != nil {
			return friendly(people.ErrNotFound)
		}
		r, err := settings.AuditEvent(q.Ctx(), c.Pool, seq)
		if db.NotFound(err) {
			return friendly(people.ErrNotFound)
		}
		if err != nil {
			return err
		}
		details, _ := json.MarshalIndent(r.Details, "", "  ")
		q.Render("identity/audit_event", "Audit event "+strconv.FormatInt(seq, 10), eventPage{Record: r, Details: string(details),
			Hash: hex.EncodeToString(r.Hash), Prev: hex.EncodeToString(r.PrevHash)})
		return nil
	})
	c.Handle("POST /console/audit/verify", authz.AuditRead, func(q *console.Request) error {
		res, err := audit.Verify(q.Ctx(), c.Pool)
		v := &verifyResult{OK: err == nil}
		if err != nil {
			v.Message = err.Error()
		} else {
			v.Message = "Verified " + strconv.FormatInt(res.Checked, 10) + " events, sequence " + strconv.FormatInt(res.From, 10) + " to " +
				strconv.FormatInt(res.To, 10) + "; every hash matches."
		}
		return p.renderAudit(q, v)
	})
}

func (p pages) renderAudit(q *console.Request, v *verifyResult) error {
	in := q.R.URL.Query()
	f := auditFilters{Action: in.Get("action"), Actor: in.Get("actor"), Target: in.Get("target"), From: in.Get("from"), To: in.Get("to")}
	before, _ := strconv.ParseInt(in.Get("before"), 10, 64)
	rows, err := settings.AuditEvents(q.Ctx(), p.c.Pool, settings.AuditFilter{Action: f.Action, Actor: f.Actor, Target: f.Target, From: day(f.From, false),
		To: day(f.To, true), Before: before, Limit: 50})
	if err != nil {
		return err
	}
	page := auditPage{Rows: rows, Filters: f, Verify: v}
	if len(rows) == 50 {
		next := url.Values{"action": {f.Action}, "actor": {f.Actor}, "target": {f.Target}, "from": {f.From}, "to": {f.To},
			"before": {strconv.FormatInt(rows[len(rows)-1].Seq, 10)}}
		page.Next = "/console/audit?" + next.Encode()
	}
	q.Render("identity/audit", "Audit log", page)
	return nil
}
