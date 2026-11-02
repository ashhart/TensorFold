package scim

import (
	"encoding/json"
	"strconv"
	"strings"
)

// AttrPath names an attribute, optionally schema-qualified, with at most one sub-attribute.
type AttrPath struct {
	URN  string
	Attr string
	Sub  string
}

// Key is the case-folded lookup key used by attribute tables.
func (p AttrPath) Key() string {
	k := strings.ToLower(p.Attr)
	if p.Sub != "" {
		k += "." + strings.ToLower(p.Sub)
	}
	if p.URN != "" {
		return strings.ToLower(p.URN) + ":" + k
	}
	return k
}

// Value is a filter comparison literal.
type Value struct {
	Kind valueKind
	Str  string
	Bool bool
}

type valueKind int

const (
	valString valueKind = iota
	valNumber
	valBool
	valNull
)

// Filter is a parsed RFC 7644 filter expression.
type Filter interface{ filterNode() }

type logical struct {
	Op   string
	L, R Filter
}

type negation struct{ X Filter }

type comparison struct {
	Path AttrPath
	Op   string
	Val  Value
}

// valuePath is attr[filter], optionally followed by .sub op value as a lenient extension.
type valuePath struct {
	Path  AttrPath
	Inner Filter
	Sub   string
	Op    string
	Val   Value
}

func (logical) filterNode()    {}
func (negation) filterNode()   {}
func (comparison) filterNode() {}
func (valuePath) filterNode()  {}

var compareOps = map[string]bool{"eq": true, "ne": true, "co": true, "sw": true, "ew": true, "gt": true, "ge": true, "lt": true, "le": true, "pr": true}

type parser struct {
	toks  []token
	i     int
	nodes int
	depth int
	inner bool
}

// ParseFilter parses a filter; any syntax it cannot read is an invalidFilter error.
func ParseFilter(s string) (Filter, error) {
	toks, err := lex(s)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: toks}
	f, err := p.or()
	if err != nil {
		return nil, err
	}
	if p.peek().kind != tokEOF {
		return nil, p.fail("unexpected %q", p.peek().text)
	}
	return f, nil
}

func (p *parser) peek() token { return p.toks[p.i] }

func (p *parser) next() token {
	t := p.toks[p.i]
	if t.kind != tokEOF {
		p.i++
	}
	return t
}

func (p *parser) fail(format string, args ...any) error {
	e := badRequest(TypeInvalidFilter, format, args...)
	e.Detail += " at position " + strconv.Itoa(p.peek().pos)
	return e
}

func (p *parser) isWord(w string) bool {
	t := p.peek()
	return t.kind == tokWord && strings.EqualFold(t.text, w)
}

func (p *parser) count() error {
	p.nodes++
	if p.nodes > maxFilterNodes {
		return badRequest(TypeInvalidFilter, "filter has more than %d terms", maxFilterNodes)
	}
	return nil
}

func (p *parser) or() (Filter, error) {
	l, err := p.and()
	if err != nil {
		return nil, err
	}
	for p.isWord("or") {
		p.next()
		r, err := p.and()
		if err != nil {
			return nil, err
		}
		l = logical{Op: "or", L: l, R: r}
	}
	return l, nil
}

func (p *parser) and() (Filter, error) {
	l, err := p.unary()
	if err != nil {
		return nil, err
	}
	for p.isWord("and") {
		p.next()
		r, err := p.unary()
		if err != nil {
			return nil, err
		}
		l = logical{Op: "and", L: l, R: r}
	}
	return l, nil
}

func (p *parser) unary() (Filter, error) {
	if err := p.count(); err != nil {
		return nil, err
	}
	p.depth++
	defer func() { p.depth-- }()
	if p.depth > maxFilterDepth {
		return nil, badRequest(TypeInvalidFilter, "filter nests deeper than %d", maxFilterDepth)
	}
	if p.isWord("not") && p.toks[p.i+1].kind == tokLParen {
		p.next()
		f, err := p.group()
		if err != nil {
			return nil, err
		}
		return negation{X: f}, nil
	}
	if p.peek().kind == tokLParen {
		return p.group()
	}
	return p.attrExpr()
}

func (p *parser) group() (Filter, error) {
	p.next()
	f, err := p.or()
	if err != nil {
		return nil, err
	}
	if p.next().kind != tokRParen {
		return nil, p.fail("missing )")
	}
	return f, nil
}

func (p *parser) attrExpr() (Filter, error) {
	t := p.next()
	if t.kind != tokWord {
		return nil, p.fail("expected an attribute name")
	}
	path, err := parseAttrPath(t.text)
	if err != nil {
		return nil, p.fail("invalid attribute path %q", t.text)
	}
	if p.peek().kind == tokLBracket {
		return p.valuePath(path)
	}
	op, val, err := p.operator()
	if err != nil {
		return nil, err
	}
	return comparison{Path: path, Op: op, Val: val}, nil
}

func (p *parser) valuePath(path AttrPath) (Filter, error) {
	if p.inner {
		return nil, p.fail("nested value filters are not allowed")
	}
	if path.Sub != "" {
		return nil, p.fail("a value filter applies to an attribute, not a sub-attribute")
	}
	p.next()
	p.inner = true
	inner, err := p.or()
	p.inner = false
	if err != nil {
		return nil, err
	}
	if p.next().kind != tokRBracket {
		return nil, p.fail("missing ]")
	}
	vp := valuePath{Path: path, Inner: inner}
	if t := p.peek(); t.kind == tokWord && strings.HasPrefix(t.text, ".") {
		p.next()
		if !validAttrName(t.text[1:]) {
			return nil, p.fail("invalid sub-attribute %q", t.text)
		}
		vp.Sub = t.text[1:]
		if vp.Op, vp.Val, err = p.operator(); err != nil {
			return nil, err
		}
	}
	return vp, nil
}

func (p *parser) operator() (string, Value, error) {
	t := p.next()
	op := strings.ToLower(t.text)
	if t.kind != tokWord || !compareOps[op] {
		return "", Value{}, p.fail("expected a comparison operator")
	}
	if op == "pr" {
		return op, Value{}, nil
	}
	v := p.next()
	switch {
	case v.kind == tokString:
		return op, Value{Kind: valString, Str: v.text}, nil
	case v.kind == tokWord && strings.EqualFold(v.text, "true"):
		return op, Value{Kind: valBool, Bool: true}, nil
	case v.kind == tokWord && strings.EqualFold(v.text, "false"):
		return op, Value{Kind: valBool}, nil
	case v.kind == tokWord && strings.EqualFold(v.text, "null"):
		return op, Value{Kind: valNull}, nil
	case v.kind == tokWord && json.Valid([]byte(v.text)) && v.text[0] != '"':
		return op, Value{Kind: valNumber, Str: v.text}, nil
	}
	return "", Value{}, p.fail("expected a value after %s", op)
}

// parseAttrPath reads [URN ":"] attr ["." sub]; the URN ends at its last colon.
func parseAttrPath(s string) (AttrPath, error) {
	var p AttrPath
	rest := s
	if len(s) > 4 && strings.EqualFold(s[:4], "urn:") {
		switch canonicalURN(s) {
		case URNEnterprise:
			return AttrPath{URN: URNEnterprise}, nil
		case URNUser, URNGroup:
			return p, badRequest(TypeInvalidPath, "a core schema URN is not an attribute")
		}
		i := strings.LastIndex(s, ":")
		p.URN, rest = canonicalURN(s[:i]), s[i+1:]
	}
	attr, sub, hasSub := strings.Cut(rest, ".")
	if !validAttrName(attr) || (hasSub && !validAttrName(sub)) {
		return p, badRequest(TypeInvalidPath, "invalid attribute path %q", s)
	}
	p.Attr, p.Sub = attr, sub
	if p.URN == URNUser || p.URN == URNGroup {
		p.URN = ""
	}
	return p, nil
}

func validAttrName(s string) bool {
	if s == "" || len(s) > 64 {
		return false
	}
	for i, c := range s {
		ok := c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || (i > 0 && (c >= '0' && c <= '9' || c == '-' || c == '_')) || (i == 0 && c == '$')
		if !ok {
			return false
		}
	}
	return true
}

func canonicalURN(s string) string {
	for _, u := range []string{URNUser, URNGroup, URNEnterprise} {
		if strings.EqualFold(s, u) {
			return u
		}
	}
	return s
}

// PatchPath is a PATCH target: attribute, optional value filter and optional sub-attribute.
type PatchPath struct {
	AttrPath
	Filter Filter
}

// ParsePatchPath reads attrPath, attrPath.sub, or attrPath[filter] optionally followed by .sub.
func ParsePatchPath(s string) (PatchPath, error) {
	open := strings.Index(s, "[")
	if open < 0 {
		ap, err := parseAttrPath(s)
		return PatchPath{AttrPath: ap}, err
	}
	end := strings.LastIndex(s, "]")
	if end < open {
		return PatchPath{}, badRequest(TypeInvalidPath, "unbalanced brackets in path %q", s)
	}
	ap, err := parseAttrPath(s[:open])
	if err != nil {
		return PatchPath{}, err
	}
	if ap.Sub != "" || ap.Attr == "" {
		return PatchPath{}, badRequest(TypeInvalidPath, "a value filter must follow an attribute in %q", s)
	}
	inner, err := parseInner(s[open+1 : end])
	if err != nil {
		return PatchPath{}, badRequest(TypeInvalidPath, "invalid value filter in path %q", s)
	}
	pp := PatchPath{AttrPath: ap, Filter: inner}
	if after := s[end+1:]; after != "" {
		if !strings.HasPrefix(after, ".") || !validAttrName(after[1:]) {
			return PatchPath{}, badRequest(TypeInvalidPath, "invalid sub-attribute after filter in %q", s)
		}
		pp.Sub = after[1:]
	}
	return pp, nil
}

func parseInner(s string) (Filter, error) {
	toks, err := lex(s)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: toks, inner: true}
	f, err := p.or()
	if err != nil {
		return nil, err
	}
	if p.peek().kind != tokEOF {
		return nil, p.fail("unexpected %q", p.peek().text)
	}
	return f, nil
}
