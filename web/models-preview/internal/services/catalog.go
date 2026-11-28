package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// SchemaVersion is the catalog format this preview reads.
const SchemaVersion = 1

// Catalog is web/tensorfold_web/data/models.json: model support and measured runs, kept apart.
type Catalog struct {
	SchemaVersion   int             `json:"schema_version"`
	Updated         string          `json:"updated"`
	HardwareClasses []HardwareClass `json:"hardware_classes"`
	Models          []Model         `json:"models"`
	Runs            []Run           `json:"runs"`
}

// HardwareClass is one filter chip: the machines a run or a recipe covers.
type HardwareClass struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

// Model is a model the engine serves, plans to serve or serves experimentally.
type Model struct {
	ID         string    `json:"id"`
	Family     string    `json:"family"`
	Name       string    `json:"name"`
	Checkpoint string    `json:"checkpoint"`
	Support    []Support `json:"support"`
}

// Support is one backend's status for a model and the recipe that serves it.
type Support struct {
	Backend   string   `json:"backend"`
	Status    string   `json:"status"`
	RecipeURL *string  `json:"recipe_url"`
	Hardware  []string `json:"hardware"`
}

// Run is one measured configuration copied from its receipt; unknown values are null.
type Run struct {
	ID              string   `json:"id"`
	ModelID         string   `json:"model_id"`
	Note            *string  `json:"note"`
	Checkpoint      *string  `json:"checkpoint"`
	Revision        *string  `json:"revision"`
	Quant           *string  `json:"quant"`
	TFVersion       *string  `json:"tf_version"`
	Hardware        string   `json:"hardware"`
	HardwareClass   string   `json:"hardware_class"`
	Backend         string   `json:"backend"`
	PromptTokens    *int     `json:"prompt_tokens"`
	ContextCapacity *int     `json:"context_capacity"`
	Slots           *int     `json:"slots"`
	Concurrency     int      `json:"concurrency"`
	Drafting        *string  `json:"drafting"`
	Sampling        *string  `json:"sampling"`
	TTFTMs          *float64 `json:"ttft_ms"`
	TTFTP95Ms       *float64 `json:"ttft_p95_ms"`
	PrefillTPS      *float64 `json:"prefill_tps"`
	DecodeTPS       *float64 `json:"decode_tps"`
	RateSource      *string  `json:"rate_source"`
	RateScope       *string  `json:"rate_scope"`
	RateWindow      *string  `json:"rate_window"`
	MeasuredAt      *string  `json:"measured_at"`
	ReceiptURL      string   `json:"receipt_url"`
	SuiteID         *string  `json:"suite_id"`
	SuiteDigest     *string  `json:"suite_digest"`
}

// Status, scope and source values the schema admits.
const (
	StatusSupported    = "supported"
	StatusExperimental = "experimental"
	StatusPlanned      = "planned"
	ScopePerStream     = "per-stream"
	ScopeAggregate     = "aggregate"
	WindowDecode       = "decode"
	WindowSteady       = "steady-state"
	WindowEndToEnd     = "end-to-end"
	SourceClient       = "client"
	SourceServer       = "server"
)

var (
	slugPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*$`)
	datePattern = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
	backends    = []string{"metal", "cuda", "hip"}
	statuses    = []string{StatusSupported, StatusExperimental, StatusPlanned}
	scopes      = []string{ScopePerStream, ScopeAggregate}
	windows     = []string{WindowDecode, WindowSteady, WindowEndToEnd}
	sources     = []string{SourceClient, SourceServer}
)

// ParseCatalog decodes body strictly (a misspelt key is an error) and checks the rules across records.
func ParseCatalog(body []byte) (Catalog, error) {
	var catalog Catalog
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&catalog); err != nil {
		return Catalog{}, fmt.Errorf("models catalog: %w", err)
	}
	if problems := CheckCatalog(catalog); len(problems) > 0 {
		return Catalog{}, fmt.Errorf("models catalog: %s", strings.Join(problems, "; "))
	}
	return catalog, nil
}

// CheckCatalog lists every rule catalog breaks; none means it may be served.
func CheckCatalog(catalog Catalog) []string {
	var problems []string
	bad := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }
	if catalog.SchemaVersion != SchemaVersion {
		bad("schema_version is %d, this preview reads %d", catalog.SchemaVersion, SchemaVersion)
	}
	if !datePattern.MatchString(catalog.Updated) {
		bad("updated %q is not a YYYY-MM-DD date", catalog.Updated)
	}
	classes := map[string]bool{}
	for _, class := range catalog.HardwareClasses {
		if !slugPattern.MatchString(class.ID) || class.Label == "" || classes[class.ID] {
			bad("hardware class %q needs a unique slug id and a label", class.ID)
		}
		classes[class.ID] = true
	}
	models := map[string]Model{}
	for _, model := range catalog.Models {
		problems = append(problems, checkModel(model, models, classes)...)
		models[model.ID] = model
	}
	runs := map[string]bool{}
	for _, run := range catalog.Runs {
		problems = append(problems, checkRun(run, runs, models, classes)...)
		runs[run.ID] = true
	}
	return problems
}

func checkModel(model Model, seen map[string]Model, classes map[string]bool) []string {
	var problems []string
	bad := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }
	if !slugPattern.MatchString(model.ID) {
		bad("model id %q is not a slug", model.ID)
	}
	if _, dup := seen[model.ID]; dup {
		bad("model id %q appears twice", model.ID)
	}
	if model.Name == "" || model.Family == "" || model.Checkpoint == "" {
		bad("model %q needs a name, a family and a checkpoint", model.ID)
	}
	if len(model.Support) == 0 {
		bad("model %q lists no backend support", model.ID)
	}
	for _, support := range model.Support {
		if !oneOf(support.Backend, backends) || !oneOf(support.Status, statuses) {
			bad("model %q: support %q/%q is not a known backend and status", model.ID, support.Backend, support.Status)
		}
		for _, class := range support.Hardware {
			if !classes[class] {
				bad("model %q: support names unknown hardware class %q", model.ID, class)
			}
		}
	}
	return problems
}

func checkRun(run Run, seen map[string]bool, models map[string]Model, classes map[string]bool) []string {
	var problems []string
	bad := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }
	if !slugPattern.MatchString(run.ID) || seen[run.ID] {
		bad("run id %q needs to be a unique slug", run.ID)
	}
	model, ok := models[run.ModelID]
	if !ok {
		bad("run %q names unknown model %q", run.ID, run.ModelID)
	}
	if !classes[run.HardwareClass] {
		bad("run %q names unknown hardware class %q", run.ID, run.HardwareClass)
	}
	if ok && supportFor(model, run.Backend, run.HardwareClass) == nil {
		bad("run %q is on %q/%q, which model %q does not list", run.ID, run.Backend, run.HardwareClass, run.ModelID)
	}
	if run.Concurrency < 1 || run.Hardware == "" || !strings.HasPrefix(run.ReceiptURL, "https://") {
		bad("run %q needs a concurrency of at least 1, a hardware name and an https receipt", run.ID)
	}
	if run.MeasuredAt != nil && !datePattern.MatchString(*run.MeasuredAt) {
		bad("run %q measured_at %q is not a YYYY-MM-DD date", run.ID, *run.MeasuredAt)
	}
	problems = append(problems, checkRate(run)...)
	return problems
}

// checkRate keeps rates comparable: a rate names its scope, and an aggregate is the steady-state total.
func checkRate(run Run) []string {
	var problems []string
	bad := func(format string) { problems = append(problems, fmt.Sprintf(format, run.ID)) }
	rated := run.DecodeTPS != nil || run.PrefillTPS != nil
	switch {
	case !rated:
	case run.RateScope == nil || !oneOf(*run.RateScope, scopes):
		bad("run %q has a rate but no rate_scope")
	case run.RateSource != nil && !oneOf(*run.RateSource, sources):
		bad("run %q has an unknown rate_source")
	case run.RateWindow != nil && !oneOf(*run.RateWindow, windows):
		bad("run %q has an unknown rate_window")
	case *run.RateScope == ScopeAggregate && (run.RateWindow == nil || *run.RateWindow != WindowSteady):
		bad("run %q: an aggregate rate must be the steady-state total (rate_window steady-state)")
	case *run.RateScope == ScopeAggregate && run.Concurrency < 2:
		bad("run %q: an aggregate rate needs two or more streams")
	}
	if !rated && (run.RateScope != nil || run.RateWindow != nil) {
		bad("run %q labels a rate it does not have")
	}
	for _, value := range []*float64{run.DecodeTPS, run.PrefillTPS, run.TTFTMs, run.TTFTP95Ms} {
		if value != nil && *value <= 0 {
			bad("run %q has a measured value that is not positive")
		}
	}
	return problems
}

// supportFor is the model's support entry for backend that covers class.
func supportFor(model Model, backend, class string) *Support {
	for index := range model.Support {
		support := &model.Support[index]
		if support.Backend == backend && oneOf(class, support.Hardware) {
			return support
		}
	}
	return nil
}

func oneOf(value string, allowed []string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}
