package gateway

import (
	"encoding/json"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

type openAIUsage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	InputTokens         int64 `json:"input_tokens"`
	OutputTokens        int64 `json:"output_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}

type anthropicUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}

// meter accumulates usage from a response body or its stream events.
type meter struct {
	protocol   string
	usage      admission.Usage
	responseID string
}

func (m *meter) fromOpenAI(u *openAIUsage) {
	if u == nil {
		return
	}
	m.usage.Prompt = u.PromptTokens + u.InputTokens
	m.usage.Completion = u.CompletionTokens + u.OutputTokens
	m.usage.Cached = u.PromptTokensDetails.CachedTokens + u.InputTokensDetails.CachedTokens
	m.usage.Observed = true
}

func (m *meter) fromAnthropic(u *anthropicUsage, final bool) {
	if u == nil {
		return
	}
	if p := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens; p > 0 {
		m.usage.Prompt, m.usage.Cached = p, u.CacheReadInputTokens
	}
	if u.OutputTokens > 0 {
		m.usage.Completion = u.OutputTokens
	}
	if final {
		m.usage.Observed = true
	}
}

// body meters a complete JSON response.
func (m *meter) body(b []byte) {
	switch m.protocol {
	case providers.Messages:
		var r struct {
			Usage *anthropicUsage `json:"usage"`
		}
		if json.Unmarshal(b, &r) == nil {
			m.fromAnthropic(r.Usage, true)
		}
	case providers.CountTokens:
		m.usage.Observed = true
	default:
		var r struct {
			ID    string       `json:"id"`
			Usage *openAIUsage `json:"usage"`
		}
		if json.Unmarshal(b, &r) == nil {
			m.fromOpenAI(r.Usage)
			if m.protocol == providers.Responses {
				m.responseID = r.ID
			}
		}
	}
}

// event meters one stream payload and reports whether it is an OpenAI usage-only chunk.
func (m *meter) event(data []byte) (usageOnly bool) {
	switch m.protocol {
	case providers.Messages:
		var e struct {
			Type    string `json:"type"`
			Message struct {
				Usage *anthropicUsage `json:"usage"`
			} `json:"message"`
			Usage *anthropicUsage `json:"usage"`
		}
		if json.Unmarshal(data, &e) != nil {
			return false
		}
		switch e.Type {
		case "message_start":
			m.fromAnthropic(e.Message.Usage, false)
		case "message_delta":
			m.fromAnthropic(e.Usage, true)
		}
	case providers.Responses:
		var e struct {
			Type     string `json:"type"`
			Response struct {
				ID    string       `json:"id"`
				Usage *openAIUsage `json:"usage"`
			} `json:"response"`
		}
		if json.Unmarshal(data, &e) != nil {
			return false
		}
		switch e.Type {
		case "response.created":
			m.responseID = e.Response.ID
		case "response.completed", "response.incomplete", "response.failed":
			m.fromOpenAI(e.Response.Usage)
		}
	default:
		var c struct {
			Choices []json.RawMessage `json:"choices"`
			Usage   *openAIUsage      `json:"usage"`
		}
		if json.Unmarshal(data, &c) != nil {
			return false
		}
		m.fromOpenAI(c.Usage)
		return c.Usage != nil && c.Choices != nil && len(c.Choices) == 0
	}
	return false
}
