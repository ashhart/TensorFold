package proxy

import (
	"net/http"
	"strings"
)

// forwarded lists the only request headers that reach an engine; credentials and cookies never do.
var forwarded = []string{"Content-Type", "Accept", "Anthropic-Version", "Anthropic-Beta", "Openai-Beta", "X-Request-Id",
	"Traceparent", "Tracestate"}

func copyRequestHeaders(dst, src http.Header) {
	for _, k := range forwarded {
		if vs := src.Values(k); len(vs) > 0 {
			dst[http.CanonicalHeaderKey(k)] = append([]string(nil), vs...)
		}
	}
}

var hopByHop = map[string]bool{"Connection": true, "Keep-Alive": true, "Proxy-Authenticate": true, "Proxy-Authorization": true,
	"Te": true, "Trailer": true, "Transfer-Encoding": true, "Upgrade": true, "Set-Cookie": true, RefusedHeader: true}

func copyResponseHeaders(dst, src http.Header) {
	drop := map[string]bool{}
	for _, v := range src.Values("Connection") {
		for _, f := range strings.Split(v, ",") {
			drop[http.CanonicalHeaderKey(strings.TrimSpace(f))] = true
		}
	}
	for k, vs := range src {
		if hopByHop[k] || drop[k] {
			continue
		}
		dst[k] = append([]string(nil), vs...)
	}
}
