package scim

import (
	"encoding/json"
	"net/http"
	"strings"
)

const (
	maxDocBytes   = 64 << 10
	maxMultiValue = 100
)

// normalize canonicalises a client document and drops read-only, write-only and protocol attributes.
func normalize(r resource, in map[string]any) (map[string]any, error) {
	out := map[string]any{}
	for k, v := range in {
		if err := put(r, out, k, v); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func put(r resource, out map[string]any, key string, v any) error {
	if strings.EqualFold(key, "schemas") || strings.EqualFold(key, "meta") {
		return nil
	}
	if len(key) < 4 || !strings.EqualFold(key[:4], "urn:") {
		return putAttr(r.attrs(), out, key, v)
	}
	switch canonicalURN(key) {
	case r.urn():
		obj, ok := v.(map[string]any)
		if !ok {
			return badRequest(TypeInvalidValue, "%s must be an object", key)
		}
		for k, sv := range obj {
			if err := put(r, out, k, sv); err != nil {
				return err
			}
		}
		return nil
	case URNEnterprise:
		obj, ok := v.(map[string]any)
		if !ok {
			return badRequest(TypeInvalidValue, "%s must be an object", URNEnterprise)
		}
		if r != resUser {
			return nil
		}
		ext := child(out, URNEnterprise)
		for k, sv := range obj {
			if err := putAttr(enterpriseAttrs, ext, k, sv); err != nil {
				return err
			}
		}
		return nil
	}
	p, err := parseAttrPath(key)
	if err != nil || p.Sub != "" {
		out[key] = v
		return nil
	}
	defs, container, known := r.defsFor(p)
	if !known {
		out[key] = v
		return nil
	}
	target := out
	if container != "" {
		target = child(out, container)
	}
	return putAttr(defs, target, p.Attr, v)
}

func child(m map[string]any, key string) map[string]any {
	if c, ok := m[key].(map[string]any); ok {
		return c
	}
	c := map[string]any{}
	m[key] = c
	return c
}

func putAttr(defs []attr, out map[string]any, key string, v any) error {
	a, ok := findAttr(defs, key)
	if !ok {
		out[key] = v
		return nil
	}
	if a.Mutability == "readOnly" || a.Mutability == "writeOnly" {
		return nil
	}
	nv, err := coerce(a, v)
	if err != nil {
		return err
	}
	if nv == nil {
		delete(out, a.Name)
		return nil
	}
	out[a.Name] = nv
	return nil
}

// coerce checks a value against its definition, accepting the loose shapes common IdPs send.
func coerce(a attr, v any) (any, error) {
	if v == nil {
		return nil, nil
	}
	if !a.Multi {
		return coerceOne(a, v)
	}
	arr, ok := v.([]any)
	if !ok {
		arr = []any{v}
	}
	if len(arr) > maxMultiValue && a.Name != "members" {
		return nil, badRequest(TypeInvalidValue, "%s has more than %d values", a.Name, maxMultiValue)
	}
	out := make([]any, 0, len(arr))
	for _, el := range arr {
		c, err := coerceOne(a, el)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, nil
}

func coerceOne(a attr, v any) (any, error) {
	switch a.Type {
	case "boolean":
		b, ok := asBool(v)
		if !ok {
			return nil, badRequest(TypeInvalidValue, "%s must be a boolean", a.Name)
		}
		return b, nil
	case "complex":
		obj, ok := v.(map[string]any)
		if !ok {
			if s, isStr := v.(string); isStr {
				if _, hasValue := findAttr(a.Sub, "value"); hasValue {
					return map[string]any{"value": s}, nil
				}
			}
			return nil, badRequest(TypeInvalidValue, "%s must be an object", a.Name)
		}
		out := map[string]any{}
		for k, sv := range obj {
			sa, ok := findAttr(a.Sub, k)
			if !ok {
				out[k] = sv
				continue
			}
			if sa.Mutability == "readOnly" || sv == nil {
				continue
			}
			c, err := coerceOne(sa, sv)
			if err != nil {
				return nil, err
			}
			out[sa.Name] = c
		}
		return out, nil
	}
	if s, ok := asString(v); ok {
		return s, nil
	}
	return nil, badRequest(TypeInvalidValue, "%s must be a string", a.Name)
}

// checkSize refuses documents that would exceed the stored attribute budget.
func checkSize(doc map[string]any) error {
	b, err := json.Marshal(doc)
	if err != nil {
		return badRequest(TypeInvalidValue, "attributes cannot be encoded")
	}
	if len(b) > maxDocBytes {
		return errf(http.StatusRequestEntityTooLarge, "", "resource attributes exceed %d bytes", maxDocBytes)
	}
	return nil
}

// project applies the attributes and excludedAttributes query parameters to a representation.
func project(rep map[string]any, attributes, excluded []string) map[string]any {
	if len(attributes) == 0 && len(excluded) == 0 {
		return rep
	}
	if len(attributes) > 0 {
		out := map[string]any{"id": rep["id"], "schemas": rep["schemas"]}
		for _, a := range attributes {
			p, err := parseAttrPath(a)
			if err != nil {
				continue
			}
			copyPath(rep, out, p)
		}
		return out
	}
	out := make(map[string]any, len(rep))
	for k, v := range rep {
		out[k] = v
	}
	for _, a := range excluded {
		p, err := parseAttrPath(a)
		if err != nil || (p.URN == "" && (strings.EqualFold(p.Attr, "id") || strings.EqualFold(p.Attr, "schemas"))) {
			continue
		}
		dropPath(out, p)
	}
	return out
}

func containerOf(m map[string]any, p AttrPath, create bool) map[string]any {
	if p.URN == "" {
		return m
	}
	if c, ok := m[p.URN].(map[string]any); ok {
		return c
	}
	if !create {
		return nil
	}
	c := map[string]any{}
	m[p.URN] = c
	return c
}

func copyPath(src, dst map[string]any, p AttrPath) {
	from := containerOf(src, p, false)
	if from == nil {
		return
	}
	if p.Attr == "" {
		dst[p.URN] = from
		return
	}
	key := keyFold(from, p.Attr)
	if key == "" {
		return
	}
	to := containerOf(dst, p, true)
	if p.Sub == "" {
		to[key] = from[key]
		return
	}
	switch v := from[key].(type) {
	case map[string]any:
		inner, _ := to[key].(map[string]any)
		if inner == nil {
			inner = map[string]any{}
		}
		if sk := keyFold(v, p.Sub); sk != "" {
			inner[sk] = v[sk]
		}
		to[key] = inner
	case []any:
		els := make([]any, 0, len(v))
		for _, el := range v {
			if obj, ok := el.(map[string]any); ok {
				if sk := keyFold(obj, p.Sub); sk != "" {
					els = append(els, map[string]any{sk: obj[sk]})
				}
			}
		}
		to[key] = els
	}
}

func dropPath(m map[string]any, p AttrPath) {
	c := containerOf(m, p, false)
	if c == nil {
		return
	}
	if p.Attr == "" {
		delete(m, p.URN)
		return
	}
	key := keyFold(c, p.Attr)
	if key == "" {
		return
	}
	if p.Sub == "" {
		if p.URN != "" {
			cp := cloneMap(c)
			delete(cp, key)
			m[p.URN] = cp
			return
		}
		delete(c, key)
		return
	}
	if obj, ok := c[key].(map[string]any); ok {
		cp := cloneMap(obj)
		delete(cp, keyFold(cp, p.Sub))
		c[key] = cp
	}
}

func cloneMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// splitList parses a comma-separated attribute list parameter.
func splitList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}
