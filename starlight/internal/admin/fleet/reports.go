package fleet

import (
	"context"
	"encoding/csv"
	"io"
	"strconv"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// UsageFilter selects usage rows; nil Projects means every project, and Limit 0 means the top 1,000 groups.
type UsageFilter struct {
	From     time.Time
	To       time.Time
	Projects []string
	GroupBy  string
	Limit    int
}

// ExportLimit bounds a CSV export, which carries every group a range holds up to this many.
const ExportLimit = 200_000

// UsageRow is one group's totals; on the overall total, Groups counts every group in the range, shown or not.
type UsageRow struct {
	Key        string
	Label      string
	Requests   int64
	Prompt     int64
	Completion int64
	Cached     int64
	CostMicros int64
	Groups     int64
}

// Cost formats the row's cost.
func (r UsageRow) Cost() string { return Dollars(r.CostMicros) }

// Tokens is prompt plus completion tokens.
func (r UsageRow) Tokens() int64 { return r.Prompt + r.Completion }

var groupings = map[string][2]string{
	"project": {"e.project_id::text", "coalesce(p.slug, e.project_id::text)"},
	"model":   {"e.model_name", "e.model_name"},
	"user":    {"e.principal_id::text", "coalesce(u.email, sa.name, e.principal_id::text)"},
	"day":     {"to_char(e.created_at, 'YYYY-MM-DD')", "to_char(e.created_at, 'YYYY-MM-DD')"},
}

// Usage totals usage by group, newest or largest first, with an overall total.
func Usage(ctx context.Context, q db.Querier, f UsageFilter) ([]UsageRow, UsageRow, error) {
	g, ok := groupings[f.GroupBy]
	if !ok {
		return nil, UsageRow{}, Invalid("group by project, model, user or day")
	}
	if !f.To.After(f.From) || f.To.Sub(f.From) > 400*24*time.Hour {
		return nil, UsageRow{}, Invalid("choose a range of at most 400 days that ends after it starts")
	}
	order := "3 DESC"
	if f.GroupBy == "day" {
		order = "1 DESC"
	}
	projects := f.Projects
	all := projects == nil
	if projects == nil {
		projects = []string{}
	}
	limit := f.Limit
	if limit <= 0 {
		limit = 1000
	}
	rows, err := q.Query(ctx, `SELECT `+g[0]+`, `+g[1]+`, count(*), sum(e.prompt_tokens)::bigint, sum(e.completion_tokens)::bigint, sum(e.cost_micros)::bigint, sum(e.cached_tokens)::bigint,
		count(*) OVER (), sum(count(*)) OVER ()::bigint, sum(sum(e.prompt_tokens)) OVER ()::bigint, sum(sum(e.completion_tokens)) OVER ()::bigint,
		sum(sum(e.cost_micros)) OVER ()::bigint, sum(sum(e.cached_tokens)) OVER ()::bigint
		FROM usage_events e LEFT JOIN projects p ON p.id = e.project_id
		LEFT JOIN users u ON e.principal_type = 'user' AND u.id = e.principal_id
		LEFT JOIN service_accounts sa ON e.principal_type = 'service_account' AND sa.id = e.principal_id
		WHERE e.created_at >= $1 AND e.created_at < $2 AND ($3 OR e.project_id = ANY($4::uuid[]))
		GROUP BY 1, 2 ORDER BY `+order+` LIMIT $5`, f.From, f.To, all, projects, limit)
	if err != nil {
		return nil, UsageRow{}, err
	}
	defer rows.Close()
	var out []UsageRow
	var total UsageRow
	for rows.Next() {
		var r UsageRow
		if err := rows.Scan(&r.Key, &r.Label, &r.Requests, &r.Prompt, &r.Completion, &r.CostMicros, &r.Cached,
			&total.Groups, &total.Requests, &total.Prompt, &total.Completion, &total.CostMicros, &total.Cached); err != nil {
			return nil, total, err
		}
		out = append(out, r)
	}
	return out, total, rows.Err()
}

// WriteCSV writes usage rows with a header; cost is in dollars with six decimals.
func WriteCSV(w io.Writer, groupBy string, rows []UsageRow) error {
	c := csv.NewWriter(w)
	if err := c.Write([]string{groupBy, "requests", "prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd"}); err != nil {
		return err
	}
	for _, r := range rows {
		cost := strconv.FormatInt(r.CostMicros/1_000_000, 10) + "." + leftPad(strconv.FormatInt(r.CostMicros%1_000_000, 10), 6)
		rec := []string{safeCell(r.Label), strconv.FormatInt(r.Requests, 10), strconv.FormatInt(r.Prompt, 10), strconv.FormatInt(r.Completion, 10),
			strconv.FormatInt(r.Cached, 10), cost}
		if err := c.Write(rec); err != nil {
			return err
		}
	}
	c.Flush()
	return c.Error()
}

// safeCell stops spreadsheet programs from treating a label as a formula.
func safeCell(s string) string {
	if s != "" && (s[0] == '=' || s[0] == '+' || s[0] == '-' || s[0] == '@' || s[0] == '\t' || s[0] == '\r') {
		return "'" + s
	}
	return s
}
