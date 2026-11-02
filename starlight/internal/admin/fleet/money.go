package fleet

import (
	"strconv"
	"strings"
)

// ParseDollars reads a non-negative dollar amount with up to six decimals into micro-dollars; empty is zero.
func ParseDollars(field, s string) (int64, error) {
	s = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(s), "$"))
	if s == "" {
		return 0, nil
	}
	whole, frac, _ := strings.Cut(s, ".")
	if whole == "" {
		whole = "0"
	}
	if len(frac) > 6 || strings.ContainsAny(whole+frac, "+-eE ") {
		return 0, Invalid(field + " must be a dollar amount with at most six decimals")
	}
	w, err := strconv.ParseInt(whole, 10, 64)
	if err != nil || w < 0 || w > 1_000_000_000 {
		return 0, Invalid(field + " must be a dollar amount with at most six decimals")
	}
	f := int64(0)
	if frac != "" {
		f, err = strconv.ParseInt(frac+strings.Repeat("0", 6-len(frac)), 10, 64)
		if err != nil {
			return 0, Invalid(field + " must be a dollar amount with at most six decimals")
		}
	}
	return w*1_000_000 + f, nil
}

// Dollars formats micro-dollars without losing precision, trimming trailing zeros beyond cents.
func Dollars(micros int64) string {
	neg := micros < 0
	if neg {
		micros = -micros
	}
	s := strconv.FormatInt(micros/1_000_000, 10) + "." + leftPad(strconv.FormatInt(micros%1_000_000, 10), 6)
	s = strings.TrimRight(s, "0")
	if i := strings.IndexByte(s, '.'); len(s)-i < 3 {
		s += strings.Repeat("0", 3-(len(s)-i))
	}
	if neg {
		return "-$" + s
	}
	return "$" + s
}

func leftPad(s string, n int) string {
	if len(s) >= n {
		return s
	}
	return strings.Repeat("0", n-len(s)) + s
}

// Count formats a whole number with thousands separators.
func Count(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}
