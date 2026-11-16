// Package validate holds every input shape; rules that need the catalog stay in the service.
package validate

import (
	"net/url"
	"regexp"
	"sort"
	"strings"
)

// Validator is implemented by every input type.
type Validator interface {
	Validate() error
}

// Error is a validation failure. Fields maps field name to message.
type Error struct {
	Fields map[string]string
}

func (err *Error) Error() string {
	keys := make([]string, 0, len(err.Fields))
	for key := range err.Fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for index, key := range keys {
		parts[index] = key + ": " + err.Fields[key]
	}
	return "invalid input: " + strings.Join(parts, ", ")
}

// Sort columns and directions of the board.
const (
	SortRate        = "rate"
	SortModel       = "model"
	SortHardware    = "hardware"
	SortTTFT        = "ttft"
	SortContext     = "context"
	SortConcurrency = "concurrency"
	SortStatus      = "status"
	SortDate        = "date"
	Ascending       = "asc"
	Descending      = "desc"
)

// SortColumns lists every column the board sorts by.
var SortColumns = []string{SortModel, SortHardware, SortRate, SortTTFT, SortContext, SortConcurrency, SortStatus, SortDate}

var slug = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]{0,63}$`)

// BoardQuery is the board's input: filter ("" is all), picked hardware, sort column, direction, open run.
type BoardQuery struct {
	Hardware string `json:"hw"`
	Mine     string `json:"mine"`
	Sort     string `json:"sort"`
	Dir      string `json:"dir"`
	Run      string `json:"run"`
}

// BoardQueryFrom reads a query string; empty values take the defaults (every hardware, fastest decode first).
func BoardQueryFrom(values url.Values) BoardQuery {
	query := BoardQuery{
		Hardware: values.Get("hw"),
		Mine:     values.Get("mine"),
		Sort:     values.Get("sort"),
		Dir:      values.Get("dir"),
		Run:      values.Get("run"),
	}
	if query.Sort == "" {
		query.Sort = SortRate
	}
	if query.Dir == "" {
		query.Dir = Descending
	}
	return query
}

// Values is the query string that reads query back, without defaults.
func (query BoardQuery) Values() url.Values {
	values := url.Values{}
	for key, value := range map[string]string{"hw": query.Hardware, "mine": query.Mine, "run": query.Run} {
		if value != "" {
			values.Set(key, value)
		}
	}
	if query.Sort != "" && query.Sort != SortRate {
		values.Set("sort", query.Sort)
	}
	if query.Dir == Ascending {
		values.Set("dir", Ascending)
	}
	return values
}

// Validate checks the shape of every field.
func (query *BoardQuery) Validate() error {
	fields := map[string]string{}
	for name, value := range map[string]string{"hw": query.Hardware, "mine": query.Mine, "run": query.Run} {
		if value != "" && !slug.MatchString(value) {
			fields[name] = "must be a lowercase slug"
		}
	}
	if query.Sort != "" && !contains(SortColumns, query.Sort) {
		fields["sort"] = "must be one of " + strings.Join(SortColumns, ", ")
	}
	if query.Dir != "" && query.Dir != Ascending && query.Dir != Descending {
		fields["dir"] = "must be asc or desc"
	}
	if len(fields) > 0 {
		return &Error{Fields: fields}
	}
	return nil
}

// ReadRun is ReadRun's input: one run's id.
type ReadRun struct {
	ID string `json:"id"`
}

// Validate checks the id is a slug.
func (input *ReadRun) Validate() error {
	if !slug.MatchString(input.ID) {
		return &Error{Fields: map[string]string{"id": "must be a lowercase slug"}}
	}
	return nil
}

func contains(list []string, value string) bool {
	for _, candidate := range list {
		if candidate == value {
			return true
		}
	}
	return false
}
