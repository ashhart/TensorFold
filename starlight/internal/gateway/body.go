// Package gateway serves the OpenAI- and Anthropic-compatible inference API under Starlight's policy.
package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Body is a request's top-level JSON object; duplicate keys are refused so policy and engine read the same values.
type Body struct {
	keys []string
	vals map[string]json.RawMessage
}

// ErrBody reports a body that is not one well-formed JSON object.
var ErrBody = errors.New("request body must be a single JSON object without duplicate keys")

// ParseBody decodes the top level of a JSON object, keeping nested values raw.
func ParseBody(b []byte) (*Body, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, ErrBody
	}
	out := &Body{vals: map[string]json.RawMessage{}}
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return nil, ErrBody
		}
		k, ok := t.(string)
		if !ok {
			return nil, ErrBody
		}
		if _, dup := out.vals[k]; dup {
			return nil, fmt.Errorf("%w: %q appears twice", ErrBody, k)
		}
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, ErrBody
		}
		out.keys = append(out.keys, k)
		out.vals[k] = v
	}
	if _, err := dec.Token(); err != nil {
		return nil, ErrBody
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, ErrBody
	}
	return out, nil
}

// Raw returns a field's raw JSON.
func (b *Body) Raw(k string) (json.RawMessage, bool) {
	v, ok := b.vals[k]
	return v, ok && string(v) != "null"
}

// String reads a string field; a present non-string is an error.
func (b *Body) String(k string) (string, bool, error) {
	v, ok := b.Raw(k)
	if !ok {
		return "", false, nil
	}
	var s string
	if err := json.Unmarshal(v, &s); err != nil {
		return "", true, fmt.Errorf("%s must be a string", k)
	}
	return s, true, nil
}

// Bool reads a boolean field.
func (b *Body) Bool(k string) (bool, bool, error) {
	v, ok := b.Raw(k)
	if !ok {
		return false, false, nil
	}
	var x bool
	if err := json.Unmarshal(v, &x); err != nil {
		return false, true, fmt.Errorf("%s must be a boolean", k)
	}
	return x, true, nil
}

// Int reads a non-negative integer field.
func (b *Body) Int(k string) (int64, bool, error) {
	v, ok := b.Raw(k)
	if !ok {
		return 0, false, nil
	}
	var n json.Number
	dec := json.NewDecoder(bytes.NewReader(v))
	dec.UseNumber()
	if err := dec.Decode(&n); err != nil {
		return 0, true, fmt.Errorf("%s must be an integer", k)
	}
	x, err := n.Int64()
	if err != nil || x < 0 {
		return 0, true, fmt.Errorf("%s must be a non-negative integer", k)
	}
	return x, true, nil
}

// Set replaces or appends a field.
func (b *Body) Set(k string, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, ok := b.vals[k]; !ok {
		b.keys = append(b.keys, k)
	}
	b.vals[k] = raw
	return nil
}

// Delete removes a field.
func (b *Body) Delete(k string) {
	if _, ok := b.vals[k]; !ok {
		return
	}
	delete(b.vals, k)
	for i, x := range b.keys {
		if x == k {
			b.keys = append(b.keys[:i], b.keys[i+1:]...)
			return
		}
	}
}

// Bytes encodes the object in its original key order.
func (b *Body) Bytes() []byte {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, k := range b.keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		kb, _ := json.Marshal(k)
		buf.Write(kb)
		buf.WriteByte(':')
		buf.Write(b.vals[k])
	}
	buf.WriteByte('}')
	return buf.Bytes()
}
