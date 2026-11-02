package scim

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"
)

const maxPatchOps = 1000

type patchOp struct {
	Op    string          `json:"op"`
	Path  string          `json:"path"`
	Value json.RawMessage `json:"value"`
}

type patchBody struct {
	Schemas    []string  `json:"schemas"`
	Operations []patchOp `json:"Operations"`
}

// decodePatch reads a PatchOp message; field names match case-insensitively as encoding/json does.
func decodePatch(body []byte) ([]patchOp, error) {
	var pb patchBody
	if err := json.Unmarshal(body, &pb); err != nil {
		return nil, badRequest(TypeInvalidSyntax, "request is not a PatchOp message")
	}
	if len(pb.Operations) == 0 {
		return nil, badRequest(TypeInvalidSyntax, "PatchOp has no Operations")
	}
	if len(pb.Operations) > maxPatchOps {
		return nil, badRequest(TypeInvalidSyntax, "PatchOp has more than %d operations", maxPatchOps)
	}
	return pb.Operations, nil
}

func decodeValue(raw json.RawMessage) (any, error) {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return nil, badRequest(TypeInvalidValue, "operation value is not JSON")
	}
	return v, nil
}

// applyPatch runs every operation against doc in order; currentID guards the read-only id.
func applyPatch(r resource, doc map[string]any, ops []patchOp, currentID string) error {
	for _, op := range ops {
		name := strings.ToLower(op.Op)
		if name != "add" && name != "replace" && name != "remove" {
			return badRequest(TypeInvalidSyntax, "op must be add, replace or remove, not %q", op.Op)
		}
		val, err := decodeValue(op.Value)
		if err != nil {
			return err
		}
		if strings.TrimSpace(op.Path) != "" {
			if err := applyOne(r, doc, name, strings.TrimSpace(op.Path), val, currentID, false); err != nil {
				return err
			}
			continue
		}
		if name == "remove" {
			return badRequest(TypeNoTarget, "remove needs a path")
		}
		obj, ok := val.(map[string]any)
		if !ok {
			return badRequest(TypeInvalidValue, "value must be an object when path is omitted")
		}
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if err := applyOne(r, doc, name, k, obj[k], currentID, true); err != nil {
				return err
			}
		}
	}
	return nil
}

func applyOne(r resource, doc map[string]any, op, path string, val any, currentID string, pathless bool) error {
	if len(path) > 4 && strings.EqualFold(path[:4], "urn:") && canonicalURN(path) == r.urn() {
		if pathless {
			obj, ok := val.(map[string]any)
			if !ok {
				return badRequest(TypeInvalidValue, "%s must be an object", r.urn())
			}
			for k, v := range obj {
				if err := applyOne(r, doc, op, k, v, currentID, true); err != nil {
					return err
				}
			}
			return nil
		}
		return badRequest(TypeInvalidPath, "a schema URN alone is not a path")
	}
	pp, err := ParsePatchPath(path)
	if err != nil {
		return err
	}
	if skip, err := guardReadOnly(r, pp, op, val, currentID, pathless); skip || err != nil {
		return err
	}
	defs, containerKey, known := r.defsFor(pp.AttrPath)
	container := doc
	if containerKey != "" {
		if op == "remove" {
			if container, _ = doc[containerKey].(map[string]any); container == nil {
				return nil
			}
		} else {
			container = child(doc, containerKey)
		}
	}
	if pp.Attr == "" {
		return applyExtension(doc, containerKey, op, val)
	}
	a, isKnown := attr{Name: pp.Attr}, false
	if known {
		a, isKnown = findAttr(defs, pp.Attr)
		if !isKnown {
			a = attr{Name: pp.Attr}
		}
	}
	key := a.Name
	if !isKnown {
		if k := keyFold(container, pp.Attr); k != "" {
			key = k
		}
	}
	_, existingArray := container[key].([]any)
	isMulti := (isKnown && a.Multi) || (!isKnown && existingArray)
	switch {
	case pp.Filter != nil:
		if !isMulti {
			return badRequest(TypeInvalidPath, "%s is not multi-valued", a.Name)
		}
		return applyFiltered(container, key, a, pp, op, val)
	case pp.Sub != "":
		if isMulti {
			return badRequest(TypeInvalidPath, "address elements of %s with a value filter", a.Name)
		}
		return applySub(container, key, a, pp.Sub, op, val)
	}
	return applyAttr(container, key, a, isMulti, op, val)
}

func guardReadOnly(r resource, pp PatchPath, op string, val any, currentID string, pathless bool) (bool, error) {
	if pp.URN != "" {
		return false, nil
	}
	switch strings.ToLower(pp.Attr) {
	case "id":
		if s, ok := val.(string); ok && op != "remove" && pp.Sub == "" && strings.EqualFold(s, currentID) {
			return true, nil
		}
		return true, badRequest(TypeMutability, "id is read-only")
	case "meta", "schemas":
		if pathless {
			return true, nil
		}
		return true, badRequest(TypeMutability, "%s is read-only", pp.Attr)
	case "password":
		return true, nil
	case "groups":
		if r == resUser {
			return true, badRequest(TypeMutability, "a user's groups are changed through the group, not the user")
		}
	}
	return false, nil
}

func applyExtension(doc map[string]any, key, op string, val any) error {
	if op == "remove" {
		delete(doc, key)
		return nil
	}
	obj, ok := val.(map[string]any)
	if !ok {
		return badRequest(TypeInvalidValue, "%s must be an object", key)
	}
	ext := map[string]any{}
	if op == "add" {
		if cur, ok := doc[key].(map[string]any); ok {
			ext = cur
		}
	}
	for k, v := range obj {
		if err := putAttr(enterpriseAttrs, ext, k, v); err != nil {
			return err
		}
	}
	doc[key] = ext
	return nil
}

func applyAttr(container map[string]any, key string, a attr, isMulti bool, op string, val any) error {
	if op == "remove" {
		cur, _ := container[key].([]any)
		if isMulti && val != nil && cur != nil {
			container[key] = removeElements(cur, asList(val))
			return nil
		}
		delete(container, key)
		return nil
	}
	nv, err := coerceLoose(a, val)
	if err != nil {
		return err
	}
	if nv == nil {
		delete(container, key)
		return nil
	}
	if op == "add" && isMulti {
		cur, _ := container[key].([]any)
		container[key] = appendUnique(cur, asList(nv))
		return nil
	}
	if op == "add" && a.Type == "complex" {
		cur, curOK := container[key].(map[string]any)
		add, addOK := nv.(map[string]any)
		if curOK && addOK {
			for k, v := range add {
				cur[k] = v
			}
			return nil
		}
	}
	container[key] = nv
	return nil
}

func applySub(container map[string]any, key string, a attr, sub, op string, val any) error {
	obj, _ := container[key].(map[string]any)
	if op == "remove" {
		if obj != nil {
			delete(obj, subKey(a, obj, sub))
		}
		return nil
	}
	if obj == nil {
		obj = map[string]any{}
		container[key] = obj
	}
	sa := subAttr(a, sub)
	nv, err := coerceLoose(sa, val)
	if err != nil {
		return err
	}
	if nv == nil {
		delete(obj, subKey(a, obj, sub))
		return nil
	}
	obj[sa.Name] = nv
	return nil
}

func applyFiltered(container map[string]any, key string, a attr, pp PatchPath, op string, val any) error {
	cur, _ := container[key].([]any)
	var hits []int
	for i, el := range cur {
		ok, err := matchElement(pp.Filter, el)
		if err != nil {
			return err
		}
		if ok {
			hits = append(hits, i)
		}
	}
	if op == "remove" {
		out := cur[:0:0]
		for i, el := range cur {
			if !contains(hits, i) {
				out = append(out, el)
			} else if pp.Sub != "" {
				if obj, ok := el.(map[string]any); ok {
					delete(obj, subKey(a, obj, pp.Sub))
				}
				out = append(out, el)
			}
		}
		container[key] = out
		return nil
	}
	if len(hits) == 0 {
		seed, ok := eqPairs(pp.Filter)
		if !ok {
			return badRequest(TypeNoTarget, "no %s value matches the filter", a.Name)
		}
		el := map[string]any{}
		for k, v := range seed {
			sa := subAttr(a, k)
			if c, err := coerceLoose(sa, v); err == nil && c != nil {
				el[sa.Name] = c
			}
		}
		cur = append(cur, el)
		hits = []int{len(cur) - 1}
	}
	for _, i := range hits {
		obj, ok := cur[i].(map[string]any)
		if !ok {
			obj = map[string]any{"value": cur[i]}
			cur[i] = obj
		}
		if pp.Sub != "" {
			sa := subAttr(a, pp.Sub)
			nv, err := coerceLoose(sa, val)
			if err != nil {
				return err
			}
			obj[sa.Name] = nv
			continue
		}
		nv, err := coerceLoose(attr{Name: a.Name, Type: "complex", Sub: a.Sub}, val)
		if err != nil {
			return err
		}
		repl, _ := nv.(map[string]any)
		if op == "replace" {
			for k := range obj {
				delete(obj, k)
			}
		}
		for k, v := range repl {
			obj[k] = v
		}
	}
	container[key] = cur
	return nil
}
