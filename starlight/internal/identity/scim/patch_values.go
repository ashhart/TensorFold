package scim

import (
	"encoding/json"
	"reflect"
	"strings"
)

// coerceLoose coerces when the attribute is known and passes unknown attributes through unchanged.
func coerceLoose(a attr, v any) (any, error) {
	if a.Type == "" {
		return v, nil
	}
	return coerce(a, v)
}

func subAttr(a attr, name string) attr {
	if s, ok := findAttr(a.Sub, name); ok {
		return s
	}
	return attr{Name: name}
}

func subKey(a attr, obj map[string]any, name string) string {
	if k := keyFold(obj, name); k != "" {
		return k
	}
	return subAttr(a, name).Name
}

func asList(v any) []any {
	if arr, ok := v.([]any); ok {
		return arr
	}
	return []any{v}
}

// elementKey identifies multi-valued elements by their value sub-attribute, case-insensitively.
func elementKey(el any) string {
	if obj, ok := el.(map[string]any); ok {
		if s, ok := asString(lookupFold(obj, "value")); ok {
			return "v:" + strings.ToLower(s)
		}
	}
	if s, ok := asString(el); ok {
		return "v:" + strings.ToLower(s)
	}
	b, _ := json.Marshal(el)
	return "j:" + string(b)
}

func appendUnique(cur, add []any) []any {
	seen := map[string]int{}
	for i, el := range cur {
		seen[elementKey(el)] = i
	}
	for _, el := range add {
		k := elementKey(el)
		if i, dup := seen[k]; dup {
			if obj, ok := cur[i].(map[string]any); ok {
				if more, ok := el.(map[string]any); ok && !reflect.DeepEqual(obj, more) {
					for mk, mv := range more {
						if !strings.EqualFold(mk, "value") {
							obj[mk] = mv
						}
					}
				}
			}
			continue
		}
		seen[k] = len(cur)
		cur = append(cur, el)
	}
	return cur
}

func removeElements(cur, drop []any) []any {
	gone := map[string]bool{}
	for _, el := range drop {
		gone[elementKey(el)] = true
	}
	out := cur[:0:0]
	for _, el := range cur {
		if !gone[elementKey(el)] {
			out = append(out, el)
		}
	}
	return out
}

func contains(xs []int, x int) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
