package ui

import (
	"encoding/json"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/web/models-preview/internal/apiclient"
	"github.com/ashhart/TensorFold/web/models-preview/internal/config"
)

const (
	// dash stands for a value the receipt does not state.
	dash = "—"
	// msPerSecond turns milliseconds into seconds.
	msPerSecond = 1000
	// thousands is the digit group size of large counts.
	thousands = 3
	// barStep is the bar width granularity (CSS classes .w-0 to .w-100).
	barStep = 5
	// receiptMinSegments is owner/repo/blob/ref/file, the shortest GitHub file path.
	receiptMinSegments = 5
	// decimal is the base integers print in.
	decimal = 10
	// floatBits is the precision of the rates as decoded from JSON.
	floatBits = 64
	// rateDecimals and secondDecimals are the digits after the point.
	rateDecimals   = 1
	secondDecimals = 2
)

// rate prints a tok/s figure as its receipt does: whole numbers bare, others to one decimal place.
func rate(value *float64) string {
	if value == nil {
		return dash
	}
	if *value == float64(int64(*value)) {
		return strconv.FormatInt(int64(*value), decimal)
	}
	return strconv.FormatFloat(*value, 'f', rateDecimals, floatBits)
}

// seconds prints milliseconds as seconds.
func seconds(value *float64) string {
	if value == nil {
		return dash
	}
	return strconv.FormatFloat(*value/msPerSecond, 'f', secondDecimals, floatBits)
}

// ttft is "p50 / p95" in seconds, or the one that is known.
func ttft(run apiclient.Run) string {
	switch {
	case run.TTFTMs == nil && run.TTFTP95Ms == nil:
		return dash
	case run.TTFTP95Ms == nil:
		return seconds(run.TTFTMs)
	default:
		return seconds(run.TTFTMs) + " / " + seconds(run.TTFTP95Ms)
	}
}

func ttftLabel(run apiclient.Run) string {
	switch {
	case run.TTFTP95Ms != nil:
		return "p50 / p95"
	case run.TTFTMs != nil:
		return "median"
	default:
		return ""
	}
}

// count prints an integer with thousands separators.
func count(value *int) string {
	if value == nil {
		return dash
	}
	digits := strconv.Itoa(*value)
	var out strings.Builder
	for index, digit := range digits {
		if index > 0 && (len(digits)-index)%thousands == 0 {
			out.WriteByte(',')
		}
		out.WriteRune(digit)
	}
	return out.String()
}

func text(value *string) string {
	if value == nil || *value == "" {
		return dash
	}
	return *value
}

// suite is the suite id and digest, or one dash when the receipt names neither.
func suite(run apiclient.Run) string {
	parts := []string{}
	for _, value := range []*string{run.SuiteID, run.SuiteDigest} {
		if value != nil && *value != "" {
			parts = append(parts, *value)
		}
	}
	if len(parts) == 0 {
		return dash
	}
	return strings.Join(parts, " ")
}

// receiptPlace names a receipt by its repository and file instead of printing the whole address.
func receiptPlace(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	segments := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	if len(segments) < receiptMinSegments || segments[2] != "blob" {
		return parsed.Host + parsed.Path
	}
	return segments[0] + "/" + segments[1] + ": " + strings.Join(segments[4:], "/")
}

// load is "slots / streams": what the server admits and what the client sent.
func load(run apiclient.Run) string {
	return count(run.Slots) + " / " + strconv.Itoa(run.Concurrency)
}

// scope says what a rate counts, in words: one stream, or the steady total of N streams (which divides by N).
func scope(run apiclient.Run) string {
	if run.DecodeTPS == nil || run.RateScope == nil {
		return "no rate in receipt"
	}
	if *run.RateScope == apiclient.ScopeAggregate {
		return "total of " + strconv.Itoa(run.Concurrency) + " · steady"
	}
	return "per stream"
}

// perStream is the steady total divided by its streams, for the inspect panel.
func perStream(run apiclient.Run) string {
	if run.DecodeTPS == nil || run.RateScope == nil || *run.RateScope != apiclient.ScopeAggregate {
		return ""
	}
	each := *run.DecodeTPS / float64(run.Concurrency)
	return strconv.FormatFloat(each, 'f', rateDecimals, floatBits)
}

func source(run apiclient.Run) string {
	if run.RateSource == nil {
		return "not stated in the receipt"
	}
	if *run.RateSource == "client" {
		return "client (tokens as delivered)"
	}
	return "server-reported"
}

func window(run apiclient.Run) string {
	if run.RateWindow == nil {
		return "not stated in the receipt"
	}
	switch *run.RateWindow {
	case apiclient.WindowSteady:
		return "steady state: only while every stream generates"
	case "decode":
		return "decode: first to last token of the stream"
	default:
		return "end to end: request sent to last token"
	}
}

// barClass is the width class of a bar, in barStep steps.
func barClass(percent int) string {
	step := (percent + barStep/2) / barStep * barStep
	return "bar-fill w-" + strconv.Itoa(step)
}

func dateOrRelease(run apiclient.Run) string {
	if run.MeasuredAt != nil {
		if day, err := time.Parse(time.DateOnly, *run.MeasuredAt); err == nil {
			return day.Format("2 Jan 2006")
		}
	}
	if run.TFVersion != nil {
		return "v" + strings.TrimPrefix(*run.TFVersion, "v")
	}
	return dash
}

func statusLabel(status string) string {
	if status == "" {
		return "Unlisted"
	}
	return strings.ToUpper(status[:1]) + status[1:]
}

func checkpoint(row apiclient.Row) string {
	if row.Run.Checkpoint != nil {
		return *row.Run.Checkpoint
	}
	return row.Model.Checkpoint
}

// href is the page URL for query (works without script).
func href(query apiclient.BoardQuery) string {
	if values := query.Values().Encode(); values != "" {
		return config.AppRoutes.Home + "?" + values
	}
	return config.AppRoutes.Home
}

// fetch loads query's board in place and keeps the address bar in step for reloads and shared links.
func fetch(query apiclient.BoardQuery) string {
	board := config.AppRoutes.Board
	if values := query.Values().Encode(); values != "" {
		board += "?" + values
	}
	return "@get(" + quote(board) + "); window.history.replaceState(null, '', " + quote(href(query)) + ")"
}

// pick is the hardware picker's Datastar action: the board for the chosen class.
func pick(query apiclient.BoardQuery) string {
	query.Mine = ""
	rest := query.Values().Encode()
	if rest != "" {
		rest += "&"
	}
	value := "encodeURIComponent(evt.target.value)"
	return "@get(" + quote(config.AppRoutes.Board+"?"+rest+"mine=") + " + " + value +
		"); window.history.replaceState(null, '', " + quote(config.AppRoutes.Home+"?"+rest+"mine=") + " + " + value + ")"
}

func with(query apiclient.BoardQuery, change func(*apiclient.BoardQuery)) apiclient.BoardQuery {
	change(&query)
	query.Run = ""
	return query
}

// sorted is query sorted by column: the same column flips direction.
func sorted(query apiclient.BoardQuery, column string) apiclient.BoardQuery {
	return with(query, func(next *apiclient.BoardQuery) {
		if next.Sort == column {
			next.Dir = map[string]string{apiclient.Ascending: apiclient.Descending}[next.Dir]
			if next.Dir == "" {
				next.Dir = apiclient.Ascending
			}
			return
		}
		next.Sort, next.Dir = column, apiclient.Descending
		if column == apiclient.SortModel || column == apiclient.SortHardware || column == apiclient.SortStatus {
			next.Dir = apiclient.Ascending
		}
	})
}

func arrow(query apiclient.BoardQuery, column string) string {
	if query.Sort != column {
		return "↕"
	}
	if query.Dir == apiclient.Ascending {
		return "↑"
	}
	return "↓"
}

func ariaSort(query apiclient.BoardQuery, column string) string {
	if query.Sort != column {
		return "none"
	}
	if query.Dir == apiclient.Ascending {
		return "ascending"
	}
	return "descending"
}

// signals is the board's local Datastar state: the run whose details are open.
func signals(query apiclient.BoardQuery) string {
	data, _ := json.Marshal(map[string]string{"_open": query.Run})
	return string(data)
}

func toggle(id string) string {
	return "$_open = $_open === " + quote(id) + " ? '' : " + quote(id)
}

func hiddenUnless(id string) string {
	return "$_open !== " + quote(id)
}

func quote(value string) string {
	data, _ := json.Marshal(value)
	return "'" + strings.ReplaceAll(strings.Trim(string(data), `"`), "'", `\'`) + "'"
}

func originLabel(fresh apiclient.Freshness) string {
	parsed, err := url.Parse(fresh.Origin)
	if err != nil || parsed.Host == "" {
		return fresh.Origin
	}
	return parsed.Host + parsed.Path
}

func readAt(fresh apiclient.Freshness) string {
	if fresh.FetchedAt.IsZero() {
		return ""
	}
	return fresh.FetchedAt.UTC().Format("2 Jan 2006 15:04 UTC")
}
