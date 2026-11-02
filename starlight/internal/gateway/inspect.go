package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
	"github.com/ashhart/TensorFold/starlight/internal/routing"
)

// Info is what policy needs from a request body.
type Info struct {
	Model        string
	Stream       bool
	MaxTokens    int64
	MaxField     string
	HasMax       bool
	Estimate     int64
	Affinity     uint64
	WantsUsage   bool
	Store        bool
	PreviousID   string
	BackgroundIn bool
	Choices      int64
}

// maxOutput and maxChoices bound what a request may reserve, so its token and cost reservations cannot overflow.
const (
	maxOutput  = 1 << 20
	maxChoices = 16
)

var maxFields = map[string][]string{
	providers.ChatCompletions: {"max_completion_tokens", "max_tokens"},
	providers.Completions:     {"max_tokens"},
	providers.Responses:       {"max_output_tokens"},
	providers.Messages:        {"max_tokens"},
}

// inspect validates the fields Starlight acts on and estimates the prompt size.
func inspect(protocol string, b *Body, raw []byte) (Info, error) {
	var in Info
	var err error
	var ok bool
	if in.Model, ok, err = b.String("model"); err != nil || !ok || in.Model == "" {
		return in, badParam("model", "model is required")
	}
	if in.Stream, _, err = b.Bool("stream"); err != nil {
		return in, badParam("stream", err.Error())
	}
	for _, f := range maxFields[protocol] {
		n, present, err := b.Int(f)
		if err != nil {
			return in, badParam(f, err.Error())
		}
		if n > maxOutput {
			return in, badParam(f, f+" must be at most "+strconv.Itoa(maxOutput))
		}
		if present && (!in.HasMax || n > in.MaxTokens) {
			in.MaxTokens, in.MaxField, in.HasMax = n, f, true
		}
	}
	in.Choices = 1
	for _, f := range []string{"n", "best_of"} {
		n, present, err := b.Int(f)
		if err != nil {
			return in, badParam(f, err.Error())
		}
		if n > maxChoices {
			return in, badParam(f, "at most "+strconv.Itoa(maxChoices)+" choices per request")
		}
		if present && n > in.Choices {
			in.Choices = n
		}
	}
	if in.MaxField == "" && len(maxFields[protocol]) > 0 {
		in.MaxField = maxFields[protocol][0]
	}
	if protocol == providers.ChatCompletions && in.Stream {
		if so, ok := b.Raw("stream_options"); ok {
			var opts struct {
				IncludeUsage bool `json:"include_usage"`
			}
			if err := json.Unmarshal(so, &opts); err != nil {
				return in, badParam("stream_options", "stream_options must be an object")
			}
			in.WantsUsage = opts.IncludeUsage
		}
	}
	if protocol == providers.Responses {
		in.Store = true
		if s, present, err := b.Bool("store"); err != nil {
			return in, badParam("store", err.Error())
		} else if present {
			in.Store = s
		}
		if in.PreviousID, _, err = b.String("previous_response_id"); err != nil {
			return in, badParam("previous_response_id", err.Error())
		}
	}
	if p, _, _ := b.String("priority"); p == "background" {
		in.BackgroundIn = true
	}
	in.Estimate = estimateTokens(raw)
	in.Affinity = routing.Affinity(prefix(b))
	return in, nil
}

func badParam(param, msg string) error {
	e := httpx.Errorf(http.StatusBadRequest, "invalid_request_error", msg)
	e.Param = param
	return e
}

// prefix gathers the leading instructions and conversation that a prompt cache would share.
func prefix(b *Body) []byte {
	var out []byte
	for _, k := range []string{"system", "instructions", "tools", "messages", "input", "prompt"} {
		if v, ok := b.Raw(k); ok {
			out = append(out, v...)
			if len(out) >= 4096 {
				return out[:4096]
			}
		}
	}
	return out
}

// estimateTokens bounds prompt tokens at a third of the text bytes, counting each inline image as 1,600 tokens.
func estimateTokens(raw []byte) int64 {
	text, images := int64(0), int64(0)
	for i := 0; i < len(raw); {
		j := bytes.Index(raw[i:], []byte(";base64,"))
		if j < 0 {
			text += int64(len(raw) - i)
			break
		}
		text += int64(j)
		end := bytes.IndexByte(raw[i+j:], '"')
		if end < 0 {
			break
		}
		images++
		i += j + end
	}
	return text/3 + images*1600 + 16
}
