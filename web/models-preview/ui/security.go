package ui

import (
	"crypto/rand"
	"encoding/base64"
	"net/http"
	"strings"

	"github.com/a-h/templ"
)

// contentSecurityPolicy allows only this origin plus the request nonce, which Datastar uses (no inline styles).
func contentSecurityPolicy(nonce string) string {
	return "default-src 'none'; " +
		"script-src 'self' 'nonce-" + nonce + "'; " +
		"style-src 'self'; " +
		"img-src 'self' data:; " +
		"font-src 'self'; " +
		"connect-src 'self'; " +
		"manifest-src 'self'; " +
		"form-action 'self'; " +
		"frame-ancestors 'none'; " +
		"base-uri 'none'; " +
		"object-src 'none'"
}

// secure sets the security headers, puts a CSP nonce on the context and keeps pages out of caches.
func secure(next http.Handler, hsts bool) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		nonce := newNonce()
		header := writer.Header()
		header.Set("Content-Security-Policy", contentSecurityPolicy(nonce))
		header.Set("X-Content-Type-Options", "nosniff")
		header.Set("X-Frame-Options", "DENY")
		header.Set("Referrer-Policy", "strict-origin")
		header.Set("Permissions-Policy", "accelerometer=(), browsing-topics=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()")
		header.Set("Cross-Origin-Opener-Policy", "same-origin")
		header.Set("Cross-Origin-Resource-Policy", "same-origin")
		header.Set("Cache-Control", "no-store")
		if hsts {
			header.Set("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
		}
		next.ServeHTTP(&noStore{ResponseWriter: writer}, request.WithContext(templ.WithNonce(request.Context(), nonce)))
	})
}

// noStore forces Cache-Control: no-store on HTML and SSE when the header is written.
type noStore struct {
	http.ResponseWriter
	done bool
}

func (store *noStore) header() {
	if store.done {
		return
	}
	store.done = true
	ct := store.Header().Get("Content-Type")
	if strings.HasPrefix(ct, "text/html") || strings.HasPrefix(ct, "text/event-stream") {
		store.Header().Set("Cache-Control", "no-store")
	}
}

func (store *noStore) WriteHeader(code int) {
	store.header()
	store.ResponseWriter.WriteHeader(code)
}

func (store *noStore) Write(data []byte) (int, error) {
	store.header()
	return store.ResponseWriter.Write(data)
}

// FlushError lets http.ResponseController flush SSE through the wrapper.
func (store *noStore) FlushError() error {
	store.header()
	return http.NewResponseController(store.ResponseWriter).Flush()
}

func (store *noStore) Unwrap() http.ResponseWriter { return store.ResponseWriter }

// nonceBytes is the CSP nonce's entropy: 144 bits, 24 base64 characters with no padding.
const nonceBytes = 18

func newNonce() string {
	data := make([]byte, nonceBytes)
	_, _ = rand.Read(data) // never fails on supported platforms (crypto/rand docs)
	return base64.RawStdEncoding.EncodeToString(data)
}

// strict answers any request mux has no route for with a bare 404.
func strict(mux *http.ServeMux) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if _, pattern := mux.Handler(request); pattern == "" {
			http.NotFound(writer, request)
			return
		}
		mux.ServeHTTP(writer, request)
	})
}
