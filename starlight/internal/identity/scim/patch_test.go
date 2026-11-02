package scim

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func ops(t *testing.T, raw string) []patchOp {
	t.Helper()
	o, err := decodePatch([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return o
}

func docOf(t *testing.T, raw string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestPatchOperations(t *testing.T) {
	cases := []struct {
		name, doc, patch, want string
	}{
		{"replace active string", `{"active":true}`, `{"Operations":[{"op":"Replace","path":"active","value":"False"}]}`, `{"active":false}`},
		{"pathless object", `{"active":true}`, `{"Operations":[{"op":"replace","value":{"active":false,"displayName":"D"}}]}`,
			`{"active":false,"displayName":"D"}`},
		{"sub attribute", `{"name":{"givenName":"A"}}`, `{"Operations":[{"op":"replace","path":"name.familyName","value":"B"}]}`,
			`{"name":{"givenName":"A","familyName":"B"}}`},
		{"dotted pathless", `{}`, `{"Operations":[{"op":"Add","value":{"name.givenName":"A","NAME.familyName":"B"}}]}`,
			`{"name":{"givenName":"A","familyName":"B"}}`},
		{"filtered sub replace", `{"emails":[{"type":"work","value":"a@x"},{"type":"home","value":"b@x"}]}`,
			`{"Operations":[{"op":"replace","path":"emails[type eq \"work\"].value","value":"c@x"}]}`,
			`{"emails":[{"type":"work","value":"c@x"},{"type":"home","value":"b@x"}]}`},
		{"filtered add creates", `{}`, `{"Operations":[{"op":"add","path":"emails[type eq \"work\"].value","value":"w@x"}]}`,
			`{"emails":[{"type":"work","value":"w@x"}]}`},
		{"add dedupes", `{"members":[{"value":"m1"}]}`, `{"Operations":[{"op":"add","path":"members","value":[{"value":"M1"},{"value":"m2"}]}]}`,
			`{"members":[{"value":"m1"},{"value":"m2"}]}`},
		{"remove by filter", `{"members":[{"value":"m1"},{"value":"m2"}]}`, `{"Operations":[{"op":"remove","path":"members[value eq \"m1\"]"}]}`,
			`{"members":[{"value":"m2"}]}`},
		{"remove by value list", `{"members":[{"value":"m1"},{"value":"m2"}]}`,
			`{"Operations":[{"op":"Remove","path":"members","value":[{"$ref":null,"value":"m2"}]}]}`, `{"members":[{"value":"m1"}]}`},
		{"remove all", `{"members":[{"value":"m1"}]}`, `{"Operations":[{"op":"remove","path":"members"}]}`, `{}`},
		{"remove missing filter is no-op", `{"members":[{"value":"m1"}]}`, `{"Operations":[{"op":"remove","path":"members[value eq \"zz\"]"}]}`,
			`{"members":[{"value":"m1"}]}`},
		{"enterprise urn path", `{}`, `{"Operations":[{"op":"replace","path":"urn:ietf:params:scim:schemas:extension:enterprise:2.0:User:department","value":"R"}]}`,
			`{"urn:ietf:params:scim:schemas:extension:enterprise:2.0:User":{"department":"R"}}`},
		{"manager as string", `{}`, `{"Operations":[{"op":"replace","path":"urn:ietf:params:scim:schemas:extension:enterprise:2.0:User:manager","value":"m"}]}`,
			`{"urn:ietf:params:scim:schemas:extension:enterprise:2.0:User":{"manager":{"value":"m"}}}`},
		{"core urn pathless", `{}`, `{"Operations":[{"op":"replace","value":{"urn:ietf:params:scim:schemas:core:2.0:User:title":"T"}}]}`,
			`{"title":"T"}`},
		{"ignores own id and password", `{"displayName":"x"}`, `{"Operations":[{"op":"replace","value":{"id":"self","password":"p","displayName":"y"}}]}`,
			`{"displayName":"y"}`},
		{"canonical case", `{}`, `{"operations":[{"op":"ADD","path":"USERNAME","value":"u"}]}`, `{"userName":"u"}`},
	}
	for _, c := range cases {
		doc := docOf(t, c.doc)
		if err := applyPatch(resUser, doc, ops(t, c.patch), "self"); err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		var want map[string]any
		_ = json.Unmarshal([]byte(c.want), &want)
		gb, _ := json.Marshal(doc)
		wb, _ := json.Marshal(want)
		if string(gb) != string(wb) {
			t.Errorf("%s:\n got %s\nwant %s", c.name, gb, wb)
		}
	}
}

func TestPatchErrors(t *testing.T) {
	cases := map[string]string{
		`{"Operations":[{"op":"move","path":"a"}]}`:                                          TypeInvalidSyntax,
		`{"Operations":[{"op":"remove"}]}`:                                                   TypeNoTarget,
		`{"Operations":[{"op":"replace","value":"x"}]}`:                                      TypeInvalidValue,
		`{"Operations":[{"op":"replace","path":"id","value":"other"}]}`:                      TypeMutability,
		`{"Operations":[{"op":"replace","path":"groups","value":[]}]}`:                       TypeMutability,
		`{"Operations":[{"op":"replace","path":"active","value":"maybe"}]}`:                  TypeInvalidValue,
		`{"Operations":[{"op":"replace","path":"emails[type co \"w\"].value","value":"x"}]}`: TypeNoTarget,
		`{"Operations":[{"op":"replace","path":"name[givenName eq \"a\"]","value":"x"}]}`:    TypeInvalidPath,
		`{"Operations":[{"op":"replace","path":"emails.value","value":"x"}]}`:                TypeInvalidPath,
		`{"Operations":[{"op":"replace","path":"bad path!","value":"x"}]}`:                   TypeInvalidPath,
	}
	for raw, want := range cases {
		err := applyPatch(resUser, map[string]any{}, ops(t, raw), "self")
		var e *Error
		if !errors.As(err, &e) || e.Type != want {
			t.Errorf("%s: got %v, want %s", raw, err, want)
		}
	}
	if _, err := decodePatch([]byte(`{"Operations":[]}`)); err == nil {
		t.Error("accepted an empty PatchOp")
	}
}

func TestNormalizeDropsReadOnlyAndCoerces(t *testing.T) {
	in := docOf(t, `{"schemas":["x"],"id":"forged","meta":{"version":"1"},"groups":[{"value":"g"}],"password":"p","USERNAME":"u",
		"emails":[{"value":"a@x","primary":"True"}],"urn:ietf:params:scim:schemas:extension:enterprise:2.0:user":{"Department":"R"}}`)
	got, err := normalize(resUser, in)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"userName": "u", "emails": []any{map[string]any{"value": "a@x", "primary": true}},
		URNEnterprise: map[string]any{"department": "R"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("normalize = %#v", got)
	}
}

func TestProjection(t *testing.T) {
	rep := map[string]any{"id": "1", "schemas": []string{URNUser}, "userName": "u", "name": map[string]any{"givenName": "A", "familyName": "B"},
		"emails": []any{map[string]any{"value": "a@x", "type": "work"}}, "meta": map[string]any{"version": "v"}}
	got := project(rep, []string{"name.givenName", "EMAILS.value"}, nil)
	want := map[string]any{"id": "1", "schemas": []string{URNUser}, "name": map[string]any{"givenName": "A"}, "emails": []any{map[string]any{"value": "a@x"}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("attributes projection = %#v", got)
	}
	ex := project(rep, nil, []string{"emails", "id", "name.familyName"})
	if _, has := ex["emails"]; has || ex["id"] != "1" || ex["name"].(map[string]any)["familyName"] != nil || ex["name"].(map[string]any)["givenName"] != "A" {
		t.Fatalf("excluded projection = %#v", ex)
	}
}

func TestOnePrimary(t *testing.T) {
	arr := []any{map[string]any{"value": "a", "primary": true}, map[string]any{"value": "b", "primary": true}, map[string]any{"value": "c"}}
	onePrimary(arr)
	if arr[0].(map[string]any)["primary"] != false || arr[1].(map[string]any)["primary"] != true {
		t.Fatalf("onePrimary = %v", arr)
	}
}
