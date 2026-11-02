// Package levels defines the ordered data classifications and service classes every policy compares.
package levels

import "fmt"

// Class is a data classification; a request may only ever raise its own.
type Class int

const (
	Public Class = iota
	Internal
	Confidential
	Restricted
)

var classNames = [...]string{"public", "internal", "confidential", "restricted"}

func (c Class) String() string {
	if c < Public || c > Restricted {
		return "invalid"
	}
	return classNames[c]
}

// ParseClass accepts a classification name.
func ParseClass(s string) (Class, error) {
	for i, n := range classNames {
		if n == s {
			return Class(i), nil
		}
	}
	return 0, fmt.Errorf("unknown classification %q", s)
}

// Max returns the stricter classification.
func Max(a, b Class) Class {
	if a > b {
		return a
	}
	return b
}

// Priority is a service class used for queue ordering and engine scheduling.
type Priority int

const (
	Batch Priority = iota
	Standard
	Interactive
)

var priorityNames = [...]string{"batch", "standard", "interactive"}

func (p Priority) String() string {
	if p < Batch || p > Interactive {
		return "invalid"
	}
	return priorityNames[p]
}

// ParsePriority accepts a service class name.
func ParsePriority(s string) (Priority, error) {
	for i, n := range priorityNames {
		if n == s {
			return Priority(i), nil
		}
	}
	return 0, fmt.Errorf("unknown priority %q", s)
}

// Rank is the queue score before ageing; classes are 1000 apart so ageing never inverts more than one class.
func (p Priority) Rank() int { return int(p) * 1000 }

// Clamp lowers p to ceiling when a caller asks for more than it is allowed.
func (p Priority) Clamp(ceiling Priority) Priority {
	if p > ceiling {
		return ceiling
	}
	return p
}
