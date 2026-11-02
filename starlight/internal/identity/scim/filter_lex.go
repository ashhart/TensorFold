package scim

import (
	"encoding/json"
	"strings"
)

type tokKind int

const (
	tokEOF tokKind = iota
	tokWord
	tokString
	tokLParen
	tokRParen
	tokLBracket
	tokRBracket
)

type token struct {
	kind tokKind
	text string
	pos  int
}

const (
	maxFilterLen   = 4096
	maxFilterNodes = 64
	maxFilterDepth = 16
)

// lex splits a filter into tokens; quoted strings are decoded with JSON escape rules.
func lex(s string) ([]token, error) {
	if len(s) > maxFilterLen {
		return nil, badRequest(TypeInvalidFilter, "filter is longer than %d characters", maxFilterLen)
	}
	var out []token
	i := 0
	for i < len(s) {
		c := s[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			i++
		case c == '(' || c == ')' || c == '[' || c == ']':
			out = append(out, token{kind: map[byte]tokKind{'(': tokLParen, ')': tokRParen, '[': tokLBracket, ']': tokRBracket}[c], text: string(c), pos: i})
			i++
		case c == '"':
			end, err := stringEnd(s, i)
			if err != nil {
				return nil, err
			}
			var v string
			if err := json.Unmarshal([]byte(s[i:end]), &v); err != nil {
				return nil, badRequest(TypeInvalidFilter, "malformed string at position %d", i)
			}
			out = append(out, token{kind: tokString, text: v, pos: i})
			i = end
		default:
			start := i
			for i < len(s) && !strings.ContainsRune(" \t\n\r()[]\"", rune(s[i])) {
				i++
			}
			out = append(out, token{kind: tokWord, text: s[start:i], pos: start})
		}
	}
	return append(out, token{kind: tokEOF, pos: len(s)}), nil
}

// stringEnd returns the index just past the closing quote of the string starting at i.
func stringEnd(s string, i int) (int, error) {
	for j := i + 1; j < len(s); j++ {
		switch s[j] {
		case '\\':
			j++
		case '"':
			return j + 1, nil
		}
	}
	return 0, badRequest(TypeInvalidFilter, "unterminated string at position %d", i)
}
