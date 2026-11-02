// Package scim serves SCIM 2.0 provisioning (RFC 7643/7644) for users and groups of one IdP connection per token.
package scim

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
)

// Schema and message URNs from RFC 7643 and RFC 7644.
const (
	URNUser         = "urn:ietf:params:scim:schemas:core:2.0:User"
	URNGroup        = "urn:ietf:params:scim:schemas:core:2.0:Group"
	URNEnterprise   = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"
	URNListResponse = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
	URNPatchOp      = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
	URNError        = "urn:ietf:params:scim:api:messages:2.0:Error"
	URNSPConfig     = "urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"
	URNResourceType = "urn:ietf:params:scim:schemas:core:2.0:ResourceType"
	URNSchema       = "urn:ietf:params:scim:schemas:core:2.0:Schema"
)

// ContentType is the SCIM media type sent on every response.
const ContentType = "application/scim+json"

// SCIM error types (RFC 7644 section 3.12).
const (
	TypeInvalidFilter = "invalidFilter"
	TypeUniqueness    = "uniqueness"
	TypeMutability    = "mutability"
	TypeInvalidSyntax = "invalidSyntax"
	TypeInvalidPath   = "invalidPath"
	TypeNoTarget      = "noTarget"
	TypeInvalidValue  = "invalidValue"
	TypeTooMany       = "tooMany"
)

// Error is a SCIM protocol error with an HTTP status and an optional scimType.
type Error struct {
	Status int
	Type   string
	Detail string
}

func (e *Error) Error() string { return e.Type + ": " + e.Detail }

func errf(status int, typ, format string, args ...any) *Error {
	return &Error{Status: status, Type: typ, Detail: fmt.Sprintf(format, args...)}
}

func badRequest(typ, format string, args ...any) *Error {
	return errf(http.StatusBadRequest, typ, format, args...)
}

var (
	errNotFound     = errf(http.StatusNotFound, "", "resource not found")
	errUnauthorized = errf(http.StatusUnauthorized, "", "missing, invalid or revoked SCIM token")
	errPrecondition = errf(http.StatusPreconditionFailed, "", "resource version does not match If-Match")
)

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", ContentType)
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// writeError sends the RFC 7644 error envelope; unexpected errors become an opaque 500.
func writeError(w http.ResponseWriter, err error) {
	var e *Error
	if !errors.As(err, &e) {
		e = errf(http.StatusInternalServerError, "", "internal error")
	}
	body := map[string]any{"schemas": []string{URNError}, "status": strconv.Itoa(e.Status), "detail": e.Detail}
	if e.Type != "" {
		body["scimType"] = e.Type
	}
	if e.Status == http.StatusUnauthorized {
		w.Header().Set("WWW-Authenticate", `Bearer realm="starlight-scim"`)
	}
	writeJSON(w, e.Status, body)
}
