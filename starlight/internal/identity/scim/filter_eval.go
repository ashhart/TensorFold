package scim

import (
	"encoding/json"
	"strconv"
	"strings"
)

// matchElement evaluates a value filter against one element of a multi-valued attribute.
func matchElement(f Filter, el any) (bool, error) {
	obj, isObj := el.(map[string]any)
	switch n := f.(type) {
	case logical:
		l, err := matchElement(n.L, el)
		if err != nil {
			return false, err
		}
		if n.Op == "and" && !l {
			return false, nil
		}
		if n.Op == "or" && l {
			return true, nil
		}
		return matchElement(n.R, el)
	case negation:
		v, err := matchElement(n.X, el)
		return !v, err
	case comparison:
		if n.Path.URN != "" || n.Path.Sub != "" {
			return false, badRequest(TypeInvalidFilter, "value filters compare sub-attributes of one element")
		}
		var got any
		if isObj {
			got = lookupFold(obj, n.Path.Attr)
		} else if strings.EqualFold(n.Path.Attr, "value") {
			got = el
		}
		return compareValue(got, n.Op, n.Val)
	default:
		return false, badRequest(TypeInvalidFilter, "unsupported value filter")
	}
}

// lookupFold finds a key case-insensitively, preferring an exact match.
func lookupFold(m map[string]any, key string) any {
	if v, ok := m[key]; ok {
		return v
	}
	for k, v := range m {
		if strings.EqualFold(k, key) {
			return v
		}
	}
	return nil
}

// keyFold returns the existing spelling of key in m, or "" when absent.
func keyFold(m map[string]any, key string) string {
	if _, ok := m[key]; ok {
		return key
	}
	for k := range m {
		if strings.EqualFold(k, key) {
			return k
		}
	}
	return ""
}

func compareValue(got any, op string, want Value) (bool, error) {
	if op == "pr" {
		return present(got), nil
	}
	if want.Kind == valNull {
		switch op {
		case "eq":
			return !present(got), nil
		case "ne":
			return present(got), nil
		}
		return false, badRequest(TypeInvalidFilter, "null only compares with eq or ne")
	}
	if want.Kind == valBool {
		b, ok := asBool(got)
		switch op {
		case "eq":
			return ok && b == want.Bool, nil
		case "ne":
			return !ok || b != want.Bool, nil
		}
		return false, badRequest(TypeInvalidFilter, "booleans only compare with eq or ne")
	}
	s, ok := asString(got)
	if !ok {
		return op == "ne", nil
	}
	return compareStrings(strings.ToLower(s), op, strings.ToLower(want.Str))
}

func compareStrings(a, op, b string) (bool, error) {
	switch op {
	case "eq":
		return a == b, nil
	case "ne":
		return a != b, nil
	case "co":
		return strings.Contains(a, b), nil
	case "sw":
		return strings.HasPrefix(a, b), nil
	case "ew":
		return strings.HasSuffix(a, b), nil
	case "gt":
		return a > b, nil
	case "ge":
		return a >= b, nil
	case "lt":
		return a < b, nil
	case "le":
		return a <= b, nil
	}
	return false, badRequest(TypeInvalidFilter, "unknown operator %q", op)
}

func present(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case string:
		return x != ""
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	return true
}

// asBool accepts JSON booleans and the "True"/"False" strings some IdPs send.
func asBool(v any) (bool, bool) {
	switch x := v.(type) {
	case bool:
		return x, true
	case string:
		b, err := strconv.ParseBool(strings.ToLower(x))
		return b, err == nil
	}
	return false, false
}

func asString(v any) (string, bool) {
	switch x := v.(type) {
	case string:
		return x, true
	case json.Number:
		return x.String(), true
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64), true
	case bool:
		return strconv.FormatBool(x), true
	}
	return "", false
}

// eqPairs returns the attribute=value pairs of a filter made only of eq comparisons joined by and.
func eqPairs(f Filter) (map[string]any, bool) {
	out := map[string]any{}
	var walk func(Filter) bool
	walk = func(f Filter) bool {
		switch n := f.(type) {
		case logical:
			return n.Op == "and" && walk(n.L) && walk(n.R)
		case comparison:
			if n.Op != "eq" || n.Path.Sub != "" || n.Path.URN != "" {
				return false
			}
			switch n.Val.Kind {
			case valString, valNumber:
				out[n.Path.Attr] = n.Val.Str
			case valBool:
				out[n.Path.Attr] = n.Val.Bool
			default:
				return false
			}
			return true
		}
		return false
	}
	if !walk(f) {
		return nil, false
	}
	return out, true
}
