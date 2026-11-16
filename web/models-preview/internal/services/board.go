package services

import (
	"cmp"
	"slices"
	"strings"

	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

// Board is one query's view: ordered runs, unmeasured support apart, and the fastest single stream.
type Board struct {
	Updated    string              `json:"updated"`
	Classes    []HardwareClass     `json:"hardware_classes"`
	Query      validate.BoardQuery `json:"query"`
	Rows       []Row               `json:"rows"`
	Unmeasured []Unmeasured        `json:"unmeasured"`
	Fastest    *Row                `json:"fastest"`
	Freshness  Freshness           `json:"freshness"`
}

// Row is one measured run with what the page shows next to it.
type Row struct {
	Run           Run    `json:"run"`
	Model         Model  `json:"model"`
	Status        string `json:"status"`
	HardwareLabel string `json:"hardware_label"`
	// BarPercent is the run's decode rate against the fastest one shown, 0 to 100.
	BarPercent int `json:"bar_percent"`
}

// Unmeasured is a model's support on a backend that has no measured run yet.
type Unmeasured struct {
	Model    Model    `json:"model"`
	Support  Support  `json:"support"`
	Hardware []string `json:"hardware"`
}

// percent is the bar's full width.
const percent = 100

// BuildBoard filters and orders catalog for query; a filter also narrows the unmeasured support.
func BuildBoard(catalog Catalog, query validate.BoardQuery) (Board, error) {
	labels := map[string]string{}
	for _, class := range catalog.HardwareClasses {
		labels[class.ID] = class.Label
	}
	for _, class := range []string{query.Hardware, query.Mine} {
		if _, known := labels[class]; class != "" && !known {
			return Board{}, ErrUnknownFilter
		}
	}
	models := map[string]Model{}
	for _, model := range catalog.Models {
		models[model.ID] = model
	}
	board := Board{Updated: catalog.Updated, Classes: catalog.HardwareClasses, Query: query}
	measured := map[string]bool{}
	var overall *Row
	for _, run := range catalog.Runs {
		measured[run.ModelID+"/"+run.Backend+"/"+run.HardwareClass] = true
		model := models[run.ModelID]
		row := Row{Run: run, Model: model, HardwareLabel: labels[run.HardwareClass]}
		if support := supportFor(model, run.Backend, run.HardwareClass); support != nil {
			row.Status = support.Status
		}
		if query.Mine != "" && run.HardwareClass == query.Mine && fasterSingleStream(row, board.Fastest) {
			fastest := row
			board.Fastest = &fastest
		}
		if fasterSingleStream(row, overall) {
			fastest := row
			overall = &fastest
		}
		if query.Hardware == "" || run.HardwareClass == query.Hardware {
			board.Rows = append(board.Rows, row)
		}
	}
	// No hardware picked yet: the card opens on the fastest single stream anywhere.
	if query.Mine == "" && overall != nil {
		board.Fastest = overall
		board.Query.Mine = overall.Run.HardwareClass
	}
	scaleBars(board.Rows)
	slices.SortStableFunc(board.Rows, rowOrder(query))
	board.Unmeasured = unmeasured(catalog.Models, measured, query.Hardware, labels)
	return board, nil
}

// fasterSingleStream reports whether row beats best per stream; aggregates never compete.
func fasterSingleStream(row Row, best *Row) bool {
	run := row.Run
	if run.DecodeTPS == nil || run.RateScope == nil || *run.RateScope != ScopePerStream {
		return false
	}
	return best == nil || *run.DecodeTPS > *best.Run.DecodeTPS
}

func scaleBars(rows []Row) {
	top := 0.0
	for _, row := range rows {
		if row.Run.DecodeTPS != nil {
			top = max(top, *row.Run.DecodeTPS)
		}
	}
	for index := range rows {
		if rate := rows[index].Run.DecodeTPS; rate != nil && top > 0 {
			rows[index].BarPercent = max(1, int(*rate/top*percent))
		}
	}
}

// unmeasured lists support with no run on its backend (and the filtered class), in catalog order.
func unmeasured(models []Model, measured map[string]bool, filter string, labels map[string]string) []Unmeasured {
	var out []Unmeasured
	for _, model := range models {
		for _, support := range model.Support {
			if filter != "" && !slices.Contains(support.Hardware, filter) {
				continue
			}
			if anyMeasured(measured, model.ID, support, filter) {
				continue
			}
			names := make([]string, 0, len(support.Hardware))
			for _, class := range support.Hardware {
				names = append(names, labels[class])
			}
			out = append(out, Unmeasured{Model: model, Support: support, Hardware: names})
		}
	}
	return out
}

func anyMeasured(measured map[string]bool, modelID string, support Support, filter string) bool {
	for _, class := range support.Hardware {
		if (filter == "" || class == filter) && measured[modelID+"/"+support.Backend+"/"+class] {
			return true
		}
	}
	return false
}

// statusOrder is the status column's order: supported, experimental, planned.
var statusOrder = []string{StatusSupported, StatusExperimental, StatusPlanned}

// rowOrder sorts by the query's column, descending unless asked; missing values sort last, ties keep catalog order.
func rowOrder(query validate.BoardQuery) func(left, right Row) int {
	sign := -1
	if query.Dir == validate.Ascending {
		sign = 1
	}
	text := func(pick func(Row) string) func(left, right Row) int {
		return func(left, right Row) int { return sign * strings.Compare(pick(left), pick(right)) }
	}
	switch query.Sort {
	case validate.SortModel:
		return text(func(row Row) string { return row.Model.Name })
	case validate.SortHardware:
		return text(func(row Row) string { return row.Run.Hardware })
	case validate.SortStatus:
		return func(left, right Row) int {
			return sign * cmp.Compare(slices.Index(statusOrder, left.Status), slices.Index(statusOrder, right.Status))
		}
	case validate.SortDate:
		return nullsLast(sign, func(row Row) *string { return row.Run.MeasuredAt })
	case validate.SortTTFT:
		return nullsLast(sign, func(row Row) *float64 { return row.Run.TTFTMs })
	case validate.SortContext:
		return nullsLast(sign, func(row Row) *int { return row.Run.ContextCapacity })
	case validate.SortConcurrency:
		return func(left, right Row) int { return sign * cmp.Compare(left.Run.Concurrency, right.Run.Concurrency) }
	default:
		return nullsLast(sign, func(row Row) *float64 { return row.Run.DecodeTPS })
	}
}

func nullsLast[Value cmp.Ordered](sign int, pick func(Row) *Value) func(left, right Row) int {
	return func(left, right Row) int {
		a, b := pick(left), pick(right)
		switch {
		case a == nil && b == nil:
			return 0
		case a == nil:
			return 1
		case b == nil:
			return -1
		}
		return sign * cmp.Compare(*a, *b)
	}
}
