//! Test root for `tensorfold mcp`: the wire and lifecycle, the cache tools, and the serve children.
test {
    _ = @import("mcp_rpc_test.zig");
    _ = @import("mcp_tools_test.zig");
    _ = @import("mcp_serve_test.zig");
}
