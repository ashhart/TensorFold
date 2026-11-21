package api

import (
	"context"
	"net/http"

	"github.com/ashhart/TensorFold/web/models-preview/internal/validate"
)

// Endpoint is one public read: Serve decodes an HTTP request, Call runs in-process, both validate first.
type Endpoint[Input any, Result any] struct {
	Method, Path string
	Decode       func(request *http.Request) Input
	Handle       func(ctx context.Context, input Input) (Result, error)
}

// Pattern is the ServeMux pattern of the endpoint.
func (endpoint Endpoint[Input, Result]) Pattern() string {
	return endpoint.Method + " " + endpoint.Path
}

// Serve is the endpoint's HTTP handler.
func (endpoint Endpoint[Input, Result]) Serve() http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		result, err := endpoint.Call(request.Context(), endpoint.Decode(request))
		if err != nil {
			writeErr(writer, request, err)
			return
		}
		WriteData(writer, http.StatusOK, result)
	})
}

// Call validates input and runs the endpoint in-process.
func (endpoint Endpoint[Input, Result]) Call(ctx context.Context, input Input) (Result, error) {
	if err := validator(&input).Validate(); err != nil {
		var zero Result
		return zero, err
	}
	return endpoint.Handle(ctx, input)
}

// validator is input as a validate.Validator; a type without one fails its first test.
func validator[Input any](input *Input) validate.Validator {
	if target, ok := any(input).(validate.Validator); ok {
		return target
	}
	panic("api: endpoint input type does not implement validate.Validator")
}
