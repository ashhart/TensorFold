//! Walk the real dispatcher's route strings, including the providers of its dynamic paths.
const std = @import("std");
const auth = @import("auth.zig");
fn open(raw: []const u8, metrics: bool) bool {
    const path = auth.routePath(raw);
    return path.len == 0 or std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/slide") or staticAsset(path) or (metrics and (std.mem.eql(u8, path, "/metrics") or std.mem.eql(u8, path, "/v1/metrics")));
}
test "auth route policy covers every dispatcher path and future private routes" {
    var walked: usize = 0;
    for ([_][:0]const u8{ @embedFile("routes.zig"), @embedFile("tokens.zig"), @embedFile("responses.zig"), @embedFile("anthropic.zig"), @embedFile("recall_routes.zig") }) |source| {
        var tokens = std.zig.Tokenizer.init(source);
        while (true) {
            const token = tokens.next();
            if (token.tag == .eof) break;
            if (token.tag != .string_literal) continue;
            const path = source[token.loc.start + 1 .. token.loc.end - 1];
            if (!std.mem.startsWith(u8, path, "/")) continue;
            walked += 1;
            for ([_]bool{ false, true }) |metrics| {
                try std.testing.expectEqual(!open(path, metrics), auth.gated(path, metrics));
                const prefixed = try std.fmt.allocPrint(std.testing.allocator, "/x{s}", .{path});
                defer std.testing.allocator.free(prefixed);
                try std.testing.expectEqual(!open(prefixed, metrics), auth.gated(prefixed, metrics));
            }
        }
    }
    try std.testing.expect(walked > 25);
    for ([_][]const u8{ "/systemone", "/v1/systemone", "/x/systemone", "/systemone/?x=1", "/future/private", "/v1/slide/learn", "/memory", "/v1/responses/id" }) |path| try std.testing.expect(auth.gated(path, true));
    for ([_][]const u8{ "", "/", "/health/", "/slide", "/slide/data.js" }) |path| try std.testing.expect(!auth.gated(path, false));
    try std.testing.expect(auth.gated("/metrics", false));
    try std.testing.expect(!auth.gated("/v1/metrics/?x=1", true));
}

test "only fixed static GET assets and GET health or opened metrics are public" {
    for ([_][]const u8{ "POST", "DELETE", "PUT", "HEAD", "get" }) |method| {
        for ([_][]const u8{ "/", "/health", "/metrics", "/v1/metrics", "/slide", "/slide/data.js" }) |path| try std.testing.expect(auth.gatedRequest(method, path, true));
    }
    for ([_][]const u8{ "/slide/unknown.js", "/slide/../systemone", "/health/../systemone", "/./systemone", "/system%6fne", "/future/private" }) |path| try std.testing.expect(auth.gated(path, true));
}

fn staticAsset(path: []const u8) bool {
    return for ([_][]const u8{ "/slide/data.js", "/slide/panel.js", "/slide/flat.js", "/slide/scene.js", "/slide/tower.js", "/slide/fx.js", "/slide/bloom.js", "/slide/layout.js" }) |asset| {
        if (std.mem.eql(u8, path, asset)) break true;
    } else false;
}
