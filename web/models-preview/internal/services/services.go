// Package services holds the catalog rules, the board view and the last good copy; it imports no transport.
package services

// Kind classifies a service error; transports map it to a status code.
type Kind int

const (
	KindInternal Kind = iota
	KindInvalid
	KindNotFound
	KindUnavailable
)

// Error is a business-rule failure whose Message is safe to show to callers.
type Error struct {
	Kind    Kind
	Message string
}

func (err *Error) Error() string { return err.Message }

// Kind sentinels: errors.Is(err, ErrNotFound) holds for every not-found error.
var (
	ErrValidation  = &Error{Kind: KindInvalid, Message: "Invalid request"}
	ErrNotFound    = &Error{Kind: KindNotFound, Message: "Not found"}
	ErrUnavailable = &Error{Kind: KindUnavailable, Message: "The models catalog is not available"}
)

// Is makes the kind sentinels match every error of their kind; named errors match only themselves.
func (err *Error) Is(target error) bool {
	switch target {
	case ErrValidation, ErrNotFound, ErrUnavailable:
		return err.Kind == target.(*Error).Kind
	}
	return false
}

// Named errors the transports and the UI match with errors.Is.
var (
	ErrRunNotFound   = &Error{Kind: KindNotFound, Message: "No run with that id"}
	ErrNoCatalogYet  = &Error{Kind: KindUnavailable, Message: "The models catalog could not be read and no earlier copy is cached"}
	ErrUnknownFilter = &Error{Kind: KindInvalid, Message: "Unknown hardware filter"}
)
