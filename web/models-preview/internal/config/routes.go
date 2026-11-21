package config

// AppRoutes are the UI's pages and Datastar reads.
var AppRoutes = struct {
	Home, Board, Static, Health string
}{
	Home:   "/",
	Board:  "/board",
	Static: "/static/",
	Health: "/healthz",
}

// APIRoutes are the JSON API's endpoints, for other sites and tools.
var APIRoutes = struct {
	Board, Catalog, Run string
}{
	Board:   "/api/v1/board",
	Catalog: "/api/v1/catalog",
	Run:     "/api/v1/runs/{id}",
}
