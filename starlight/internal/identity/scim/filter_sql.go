package scim

import (
	"strconv"
	"strings"
	"time"
)

type colKind int

const (
	colText colKind = iota
	colExact
	colBool
	colTime
	colID
	colArray
	colMembers
)

// column maps a filterable attribute to SQL; for arrays, expr is the jsonb array and elems the element columns.
type column struct {
	kind  colKind
	expr  string
	extra string
	elems map[string]elemCol
}

type elemCol struct {
	expr string
	bool bool
}

func docText(path ...string) string {
	return "(u.scim_attributes #>> '{" + strings.Join(path, ",") + "}')"
}

func docArray(name string) string {
	return "(CASE WHEN jsonb_typeof(u.scim_attributes->'" + name + "') = 'array' THEN u.scim_attributes->'" + name + "' ELSE '[]'::jsonb END)"
}

// displayExpr mirrors userDoc: stored attributes when present, else the display_name column.
const displayExpr = "(CASE WHEN u.scim_attributes = '{}'::jsonb THEN nullif(u.display_name, '') ELSE u.scim_attributes->>'displayName' END)"

var plainElems = map[string]elemCol{
	"value": {expr: "(e->>'value')"}, "display": {expr: "(e->>'display')"}, "type": {expr: "(e->>'type')"},
	"primary": {expr: "(e->>'primary')", bool: true},
}

var userColumns = map[string]column{
	"id":                {kind: colID, expr: "u.id::text"},
	"username":          {kind: colText, expr: "coalesce(u.scim_user_name, u.email)"},
	"externalid":        {kind: colExact, expr: "u.external_id"},
	"displayname":       {kind: colText, expr: displayExpr},
	"active":            {kind: colBool, expr: "u.status = 'active'"},
	"emails":            {kind: colArray, expr: docArray("emails"), extra: "u.email", elems: plainElems},
	"phonenumbers":      {kind: colArray, expr: docArray("phoneNumbers"), elems: plainElems},
	"addresses":         {kind: colArray, expr: docArray("addresses"), elems: plainElems},
	"name.givenname":    {kind: colText, expr: docText("name", "givenName")},
	"name.familyname":   {kind: colText, expr: docText("name", "familyName")},
	"name.formatted":    {kind: colText, expr: docText("name", "formatted")},
	"title":             {kind: colText, expr: docText("title")},
	"usertype":          {kind: colText, expr: docText("userType")},
	"nickname":          {kind: colText, expr: docText("nickName")},
	"preferredlanguage": {kind: colText, expr: docText("preferredLanguage")},
	"locale":            {kind: colText, expr: docText("locale")},
	"meta.created":      {kind: colTime, expr: "u.created_at"},
	"meta.lastmodified": {kind: colTime, expr: "u.updated_at"},
}

var groupColumns = map[string]column{
	"id":                {kind: colID, expr: "g.id::text"},
	"displayname":       {kind: colText, expr: "g.display_name"},
	"externalid":        {kind: colExact, expr: "nullif(g.external_id, g.id::text)"},
	"members":           {kind: colMembers, elems: memberElems},
	"meta.created":      {kind: colTime, expr: "g.created_at"},
	"meta.lastmodified": {kind: colTime, expr: "g.updated_at"},
}

func init() {
	for _, a := range []string{"employeeNumber", "costCenter", "organization", "division", "department"} {
		userColumns[strings.ToLower(URNEnterprise)+":"+strings.ToLower(a)] = column{kind: colText, expr: docText(URNEnterprise, a)}
	}
	userColumns[strings.ToLower(URNEnterprise)+":manager.value"] = column{kind: colExact, expr: docText(URNEnterprise, "manager", "value")}
}

var memberElems = map[string]elemCol{
	"value": {expr: "gm.user_id::text"}, "display": {expr: "coalesce(nullif(mu.display_name, ''), mu.email)"}, "type": {expr: "'User'"},
}

const memberExists = "EXISTS (SELECT 1 FROM group_members gm JOIN users mu ON mu.id = gm.user_id WHERE gm.group_id = g.id AND mu.status <> 'deactivated'"

type compiler struct {
	cols map[string]column
	args []any
	base int
}

// compileFilter turns a filter into a parameterised WHERE fragment whose placeholders start after base.
func compileFilter(r resource, f Filter, base int) (string, []any, error) {
	c := &compiler{cols: userColumns, base: base}
	if r == resGroup {
		c.cols = groupColumns
	}
	sql, err := c.node(f)
	return sql, c.args, err
}

func (c *compiler) arg(v any) string {
	c.args = append(c.args, v)
	return "$" + strconv.Itoa(c.base+len(c.args))
}

func (c *compiler) node(f Filter) (string, error) {
	switch n := f.(type) {
	case logical:
		l, err := c.node(n.L)
		if err != nil {
			return "", err
		}
		r, err := c.node(n.R)
		if err != nil {
			return "", err
		}
		return "(" + l + " " + strings.ToUpper(n.Op) + " " + r + ")", nil
	case negation:
		x, err := c.node(n.X)
		if err != nil {
			return "", err
		}
		return "(NOT coalesce(" + x + ", false))", nil
	case comparison:
		return c.compare(n)
	case valuePath:
		return c.valuePath(n)
	}
	return "", badRequest(TypeInvalidFilter, "unsupported filter")
}

func (c *compiler) column(p AttrPath) (column, error) {
	if col, ok := c.cols[p.Key()]; ok {
		return col, nil
	}
	if strings.EqualFold(p.Sub, "value") {
		base := p
		base.Sub = ""
		if col, ok := c.cols[base.Key()]; ok && (col.kind == colArray || col.kind == colMembers) {
			return col, nil
		}
	}
	return column{}, badRequest(TypeInvalidFilter, "filtering on %q is not supported", p.Key())
}

func (c *compiler) compare(n comparison) (string, error) {
	col, err := c.column(n.Path)
	if err != nil {
		return "", err
	}
	switch col.kind {
	case colText, colExact, colID:
		return c.text(col.expr, col.kind != colText, n.Op, n.Val)
	case colBool:
		return c.boolean(col.expr, n.Op, n.Val)
	case colTime:
		return c.timestamp(col.expr, n.Op, n.Val)
	case colArray:
		op := n.Op
		if op == "ne" {
			op = "eq"
		}
		inner, err := c.text("(e->>'value')", false, op, n.Val)
		if err != nil {
			return "", err
		}
		sql := c.exists(col, inner)
		if col.extra != "" {
			extra, err := c.text(col.extra, false, op, n.Val)
			if err != nil {
				return "", err
			}
			sql = "(" + sql + " OR " + extra + ")"
		}
		if n.Op == "ne" {
			return "(NOT coalesce(" + sql + ", false))", nil
		}
		return sql, nil
	case colMembers:
		switch n.Op {
		case "pr":
			return memberExists + ")", nil
		case "eq", "ne":
			if n.Val.Kind != valString {
				return "", badRequest(TypeInvalidFilter, "members compare with a string id")
			}
			sql := memberExists + " AND gm.user_id::text = lower(" + c.arg(n.Val.Str) + "))"
			if n.Op == "ne" {
				return "(NOT " + sql + ")", nil
			}
			return sql, nil
		}
		return "", badRequest(TypeInvalidFilter, "members support eq, ne and pr")
	}
	return "", badRequest(TypeInvalidFilter, "unsupported attribute")
}

func (c *compiler) exists(col column, cond string) string {
	return "EXISTS (SELECT 1 FROM jsonb_array_elements(" + col.expr + ") AS e WHERE " + cond + ")"
}

func (c *compiler) valuePath(n valuePath) (string, error) {
	col, ok := c.cols[n.Path.Key()]
	if !ok || (col.kind != colArray && col.kind != colMembers) {
		return "", badRequest(TypeInvalidFilter, "%q does not take a value filter", n.Path.Key())
	}
	inner, err := c.element(col, n.Inner)
	if err != nil {
		return "", err
	}
	if n.Sub != "" {
		ec, ok := col.elems[strings.ToLower(n.Sub)]
		if !ok {
			return "", badRequest(TypeInvalidFilter, "unknown sub-attribute %q", n.Sub)
		}
		extra, err := c.elementCompare(ec, n.Op, n.Val)
		if err != nil {
			return "", err
		}
		inner = "(" + inner + " AND " + extra + ")"
	}
	if col.kind == colMembers {
		return memberExists + " AND " + inner + ")", nil
	}
	return c.exists(col, inner), nil
}

func (c *compiler) element(col column, f Filter) (string, error) {
	switch n := f.(type) {
	case logical:
		l, err := c.element(col, n.L)
		if err != nil {
			return "", err
		}
		r, err := c.element(col, n.R)
		if err != nil {
			return "", err
		}
		return "(" + l + " " + strings.ToUpper(n.Op) + " " + r + ")", nil
	case negation:
		x, err := c.element(col, n.X)
		if err != nil {
			return "", err
		}
		return "(NOT coalesce(" + x + ", false))", nil
	case comparison:
		if n.Path.URN != "" || n.Path.Sub != "" {
			return "", badRequest(TypeInvalidFilter, "value filters compare sub-attributes of one element")
		}
		ec, ok := col.elems[strings.ToLower(n.Path.Attr)]
		if !ok {
			return "", badRequest(TypeInvalidFilter, "unknown sub-attribute %q", n.Path.Attr)
		}
		return c.elementCompare(ec, n.Op, n.Val)
	}
	return "", badRequest(TypeInvalidFilter, "unsupported value filter")
}

func (c *compiler) elementCompare(ec elemCol, op string, v Value) (string, error) {
	if ec.bool {
		return c.boolean("lower("+ec.expr+") = 'true'", op, v)
	}
	return c.text(ec.expr, false, op, v)
}

func (c *compiler) text(expr string, exact bool, op string, v Value) (string, error) {
	if op == "pr" {
		return "(" + expr + " IS NOT NULL AND " + expr + " <> '')", nil
	}
	if v.Kind == valNull {
		switch op {
		case "eq":
			return "(" + expr + " IS NULL OR " + expr + " = '')", nil
		case "ne":
			return "(" + expr + " IS NOT NULL AND " + expr + " <> '')", nil
		}
		return "", badRequest(TypeInvalidFilter, "null only compares with eq or ne")
	}
	if v.Kind == valBool {
		return "", badRequest(TypeInvalidFilter, "a string attribute cannot be compared with a boolean")
	}
	lhs, val := "lower("+expr+")", strings.ToLower(v.Str)
	if exact {
		lhs, val = expr, v.Str
	}
	switch op {
	case "eq":
		return "(" + lhs + " = " + c.arg(val) + ")", nil
	case "ne":
		return "(" + lhs + " IS DISTINCT FROM " + c.arg(val) + ")", nil
	case "co":
		return "(" + lhs + " LIKE " + c.arg("%"+escapeLike(val)+"%") + " ESCAPE '\\')", nil
	case "sw":
		return "(" + lhs + " LIKE " + c.arg(escapeLike(val)+"%") + " ESCAPE '\\')", nil
	case "ew":
		return "(" + lhs + " LIKE " + c.arg("%"+escapeLike(val)) + " ESCAPE '\\')", nil
	case "gt", "ge", "lt", "le":
		sym := map[string]string{"gt": ">", "ge": ">=", "lt": "<", "le": "<="}[op]
		return "(" + lhs + " " + sym + " " + c.arg(val) + ")", nil
	}
	return "", badRequest(TypeInvalidFilter, "unknown operator %q", op)
}

func (c *compiler) boolean(expr, op string, v Value) (string, error) {
	if op == "pr" {
		return "TRUE", nil
	}
	want, ok := v.Bool, v.Kind == valBool
	if v.Kind == valString {
		want, ok = asBool(v.Str)
	}
	if !ok || (op != "eq" && op != "ne") {
		return "", badRequest(TypeInvalidFilter, "a boolean attribute compares with eq or ne and true or false")
	}
	if (op == "eq") == want {
		return "(" + expr + ")", nil
	}
	return "(NOT coalesce(" + expr + ", false))", nil
}

func (c *compiler) timestamp(expr, op string, v Value) (string, error) {
	if op == "pr" {
		return "TRUE", nil
	}
	if v.Kind != valString {
		return "", badRequest(TypeInvalidFilter, "a dateTime attribute compares with an RFC 3339 string")
	}
	t, err := time.Parse(time.RFC3339Nano, v.Str)
	if err != nil {
		return "", badRequest(TypeInvalidFilter, "%q is not an RFC 3339 dateTime", v.Str)
	}
	sym, ok := map[string]string{"eq": "=", "ne": "<>", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[op]
	if !ok {
		return "", badRequest(TypeInvalidFilter, "dateTime attributes do not support %s", op)
	}
	return "(" + expr + " " + sym + " " + c.arg(t) + ")", nil
}

func escapeLike(s string) string {
	return strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`).Replace(s)
}
