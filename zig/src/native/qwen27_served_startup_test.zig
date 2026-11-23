const lb = @import("qwen27_lanes.zig");

test "qwen27 startup: the adaptive block hooks are analyzed" {
    _ = &lb.Metal.init;
    _ = &lb.Metal.backend;
    _ = &lb.Metal.lone;
}

test {
    _ = @import("qwen27_startup_line.zig");
}
