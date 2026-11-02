package scim

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestParseFilterShapes(t *testing.T) {
	cases := map[string]Filter{
		`userName eq "a@example.com"`: comparison{Path: AttrPath{Attr: "userName"}, Op: "eq", Val: Value{Str: "a@example.com"}},
		`USERNAME EQ "x"`:             comparison{Path: AttrPath{Attr: "USERNAME"}, Op: "eq", Val: Value{Str: "x"}},
		`title pr`:                    comparison{Path: AttrPath{Attr: "title"}, Op: "pr"},
		`active eq True`:              comparison{Path: AttrPath{Attr: "active"}, Op: "eq", Val: Value{Kind: valBool, Bool: true}},
		`name.givenName sw "A\"b"`:    comparison{Path: AttrPath{Attr: "name", Sub: "givenName"}, Op: "sw", Val: Value{Str: `A"b`}},
		`urn:ietf:params:scim:schemas:extension:enterprise:2.0:User:manager.value eq "m"`: comparison{
			Path: AttrPath{URN: URNEnterprise, Attr: "manager", Sub: "value"}, Op: "eq", Val: Value{Str: "m"}},
		`urn:ietf:params:scim:schemas:core:2.0:User:userName eq "u"`: comparison{Path: AttrPath{Attr: "userName"}, Op: "eq", Val: Value{Str: "u"}},
		`a eq "1" or b eq "2" and c eq "3"`: logical{Op: "or",
			L: comparison{Path: AttrPath{Attr: "a"}, Op: "eq", Val: Value{Str: "1"}},
			R: logical{Op: "and", L: comparison{Path: AttrPath{Attr: "b"}, Op: "eq", Val: Value{Str: "2"}},
				R: comparison{Path: AttrPath{Attr: "c"}, Op: "eq", Val: Value{Str: "3"}}}},
		`not (a eq null)`: negation{X: comparison{Path: AttrPath{Attr: "a"}, Op: "eq", Val: Value{Kind: valNull}}},
		`emails[type eq "work" and value co "@example.com"]`: valuePath{Path: AttrPath{Attr: "emails"}, Inner: logical{Op: "and",
			L: comparison{Path: AttrPath{Attr: "type"}, Op: "eq", Val: Value{Str: "work"}},
			R: comparison{Path: AttrPath{Attr: "value"}, Op: "co", Val: Value{Str: "@example.com"}}}},
		`emails[type eq "work"].value eq "x"`: valuePath{Path: AttrPath{Attr: "emails"}, Inner: comparison{Path: AttrPath{Attr: "type"}, Op: "eq",
			Val: Value{Str: "work"}}, Sub: "value", Op: "eq", Val: Value{Str: "x"}},
		`meta.lastModified gt "2026-01-01T00:00:00Z"`: comparison{Path: AttrPath{Attr: "meta", Sub: "lastModified"}, Op: "gt", Val: Value{Str: "2026-01-01T00:00:00Z"}},
		`x eq 42`: comparison{Path: AttrPath{Attr: "x"}, Op: "eq", Val: Value{Kind: valNumber, Str: "42"}},
	}
	for in, want := range cases {
		got, err := ParseFilter(in)
		if err != nil {
			t.Errorf("%s: %v", in, err)
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s:\n got %#v\nwant %#v", in, got, want)
		}
	}
}

func TestParseFilterRejects(t *testing.T) {
	for _, in := range []string{
		``, `userName`, `userName eq`, `userName xx "a"`, `(userName eq "a"`, `userName eq "a")`, `userName eq "unterminated`,
		`emails[type eq "work"`, `emails[type[x eq "y"] eq "z"]`, `a.b.c eq "x"`, `1abc eq "x"`, `userName eq "a" and`, `not userName eq "a"`,
		`emails.value[type eq "x"]`, strings.Repeat("(", 40) + `a eq "b"` + strings.Repeat(")", 40), strings.Repeat(`a eq "b" or `, 80) + `a eq "b"`,
	} {
		_, err := ParseFilter(in)
		var e *Error
		if err == nil || !errors.As(err, &e) || e.Status != 400 || (e.Type != TypeInvalidFilter && e.Type != TypeInvalidPath) {
			t.Errorf("ParseFilter(%.40q) = %v", in, err)
		}
	}
}

func TestCompileFilterSQL(t *testing.T) {
	cases := []struct {
		r    resource
		in   string
		sql  string
		args []any
	}{
		{resUser, `userName eq "A@Example.com"`, `(lower(coalesce(u.scim_user_name, u.email)) = $2)`, []any{"a@example.com"}},
		{resUser, `externalId eq "Ab"`, `(u.external_id = $2)`, []any{"Ab"}},
		{resUser, `active eq false`, `(NOT coalesce(u.status = 'active', false))`, nil},
		{resUser, `displayName co "50%_"`, `(lower(` + displayExpr + `) LIKE $2 ESCAPE '\')`, []any{`%50\%\_%`}},
		{resGroup, `members eq "0190F5A0-0000-7000-8000-000000000000"`, memberExists + ` AND gm.user_id::text = lower($2))`, []any{"0190F5A0-0000-7000-8000-000000000000"}},
	}
	for _, c := range cases {
		f, err := ParseFilter(c.in)
		if err != nil {
			t.Fatal(err)
		}
		sql, args, err := compileFilter(c.r, f, 1)
		if err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		if sql != c.sql || !reflect.DeepEqual(args, c.args) {
			t.Errorf("%s:\n got %s %v\nwant %s %v", c.in, sql, args, c.sql, c.args)
		}
	}
	for _, in := range []string{`password eq "x"`, `userName eq true`, `active co "t"`, `meta.created eq "yesterday"`, `members co "x"`, `title[type eq "x"]`} {
		f, err := ParseFilter(in)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := compileFilter(resUser, f, 1); err == nil {
			if _, _, err := compileFilter(resGroup, f, 1); err == nil {
				t.Errorf("compiled unsupported %q", in)
			}
		}
	}
}

func FuzzParseFilter(f *testing.F) {
	for _, s := range []string{`userName eq "a"`, `emails[type eq "work"].value co "x"`, `not (a pr) or b ne null`, `urn:x:y:z eq 1`, `(`, `"`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		parsed, err := ParseFilter(s)
		if err != nil {
			var e *Error
			if !errors.As(err, &e) || e.Status != 400 {
				t.Fatalf("non-SCIM error %v", err)
			}
			return
		}
		for _, r := range []resource{resUser, resGroup} {
			sql, args, err := compileFilter(r, parsed, 1)
			if err != nil {
				continue
			}
			if strings.Count(sql, "$") < len(args) {
				t.Fatalf("placeholders %q do not cover %d args", sql, len(args))
			}
			if strings.Contains(sql, "\"") || strings.Count(sql, "'")%2 != 0 {
				t.Fatalf("a literal reached the SQL text: %q", sql)
			}
		}
	})
}
