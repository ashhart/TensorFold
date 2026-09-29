const std = @import("std");
pub const Anchor = struct { path: []const u8, symbol: []const u8, kind: enum { function, @"test" } = .function };
pub const Check = struct { step: []const u8, driver: Anchor, scope: enum { host, synthetic, checkpoint, hardware } };
pub const Feature = struct {
    id: []const u8,
    state: enum { present, partial, missing, excluded },
    requirement: []const u8,
    upstream: []const []const u8,
    implementation: []const Anchor = &.{},
    checks: []const Check = &.{},
    limitation: []const u8 = "",
};

fn localPath(path: []const u8) bool {
    if (path.len == 0 or std.fs.path.isAbsolute(path) or std.mem.indexOfScalar(u8, path, '\\') != null) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    return true;
}

fn hasDeclaration(a: std.mem.Allocator, source: [:0]const u8, value: Anchor) !bool {
    var tree = try std.zig.Ast.parse(a, source, .{ .mode = .zig });
    defer tree.deinit(a);
    if (tree.errors.len != 0) return error.InvalidZigSource;
    for (0..tree.nodes.len) |index| {
        const node: std.zig.Ast.Node.Index = @fromBackingInt(@intCast(index));
        if (value.kind == .@"test") {
            if (tree.nodeTag(node) != .test_decl) continue;
            const token = tree.nodeMainToken(node) + 1;
            if (tree.tokenTag(token) != .string_literal) continue;
            const name = try std.zig.string_literal.parseAlloc(a, tree.tokenSlice(token));
            defer a.free(name);
            if (std.mem.eql(u8, name, value.symbol)) return true;
            continue;
        }
        if (tree.nodeTag(node) != .fn_decl) continue;
        var buffer: [1]std.zig.Ast.Node.Index = undefined;
        const prototype = tree.fullFnProto(&buffer, node).?;
        if (prototype.name_token) |token| if (std.mem.eql(u8, tree.tokenSlice(token), value.symbol)) return true;
    }
    return false;
}

fn anchor(a: std.mem.Allocator, io: std.Io, value: Anchor) !void {
    if (!localPath(value.path) or !std.mem.endsWith(u8, value.path, ".zig") or value.symbol.len == 0) return error.InvalidCoverageAnchor;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, value.path, a, .limited(8 * 1024 * 1024));
    defer a.free(bytes);
    const source = try a.dupeSentinel(u8, bytes, 0);
    defer a.free(source);
    if (!try hasDeclaration(a, source, value)) {
        std.debug.print("Missing coverage declaration: {s}:{s}\n", .{ value.path, value.symbol });
        return error.MissingCoverageDeclaration;
    }
}

fn validate(features: []const Feature, sources: []const []const u8, steps: []const []const u8, report: bool) !usize {
    var incomplete: usize = 0;
    for (features, 0..) |feature, index| {
        if (feature.id.len == 0 or feature.requirement.len == 0 or feature.upstream.len == 0) return error.InvalidFeature;
        for (features[0..index]) |prior| if (std.mem.eql(u8, prior.id, feature.id)) return error.DuplicateFeature;
        if (feature.state != .present and feature.limitation.len == 0) return error.MissingCoverageLimitation;
        if (feature.state == .present and (feature.implementation.len == 0 or feature.checks.len == 0)) return error.UnbackedFeature;
        if (feature.state == .missing or feature.state == .partial) {
            incomplete += 1;
            if (report) std.debug.print("OPEN {s}: {s}\n", .{ feature.id, feature.limitation });
        }
        for (feature.upstream, 0..) |path, i| {
            if (!localPath(path)) return error.InvalidCoveragePath;
            if (!contains(sources, path)) {
                if (report) std.debug.print("Stale feature source: {s}: {s}\n", .{ feature.id, path });
                return error.StaleFeatureSource;
            }
            if (contains(feature.upstream[0..i], path)) return error.DuplicateFeatureSource;
        }
        for (feature.checks) |verification| if (!contains(steps, verification.step)) {
            if (report) std.debug.print("Missing executable test step: {s}: {s}\n", .{ feature.id, verification.step });
            return error.MissingCoverageTest;
        };
    }
    for (sources) |path| {
        var mapped = false;
        for (features) |feature| mapped = mapped or contains(feature.upstream, path);
        if (!mapped) {
            if (report) std.debug.print("Unmapped upstream source: {s}\n", .{path});
            return error.UnmappedUpstreamFeature;
        }
    }
    return incomplete;
}

fn contains(values: []const []const u8, text: []const u8) bool {
    for (values) |value| if (std.mem.eql(u8, value, text)) return true;
    return false;
}

pub fn check(a: std.mem.Allocator, io: std.Io, sources: []const []const u8, steps: []const []const u8, complete: bool) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, "native/features.json", a, .limited(4 * 1024 * 1024));
    defer a.free(bytes);
    const parsed = try std.json.parseFromSlice([]const Feature, a, bytes, .{});
    defer parsed.deinit();
    const incomplete = try validate(parsed.value, sources, steps, true);
    for (parsed.value) |feature| {
        for (feature.implementation) |value| try anchor(a, io, value);
        for (feature.checks) |value| try anchor(a, io, value.driver);
        if (feature.state == .present and feature.limitation.len > 0) std.debug.print("LIMIT {s}: {s}\n", .{ feature.id, feature.limitation });
    }
    std.debug.print("Feature bindings valid: {d} features, {d} upstream files, {d} incomplete. Bindings do not prove test execution or numerical parity.\n", .{ parsed.value.len, sources.len, incomplete });
    if (complete and incomplete != 0) return error.IncompleteNativeParity;
}

test "feature mapping rejects unassigned sources, stale bindings and unsupported completion claims" {
    const source = "src/tensorfold/model.py";
    const valid = Feature{ .id = "model", .state = .present, .requirement = "Run the model", .upstream = &.{source}, .implementation = &.{.{ .path = "native/model.zig", .symbol = "forward" }}, .checks = &.{.{ .step = "test-model", .driver = .{ .path = "native/model.zig", .symbol = "check" }, .scope = .synthetic }} };
    try std.testing.expectEqual(@as(usize, 0), try validate(&.{valid}, &.{source}, &.{"test-model"}, false));
    try std.testing.expectError(error.UnmappedUpstreamFeature, validate(&.{valid}, &.{ source, "src/tensorfold/new.py" }, &.{"test-model"}, false));
    try std.testing.expectError(error.StaleFeatureSource, validate(&.{valid}, &.{}, &.{"test-model"}, false));
    try std.testing.expectError(error.MissingCoverageTest, validate(&.{valid}, &.{source}, &.{}, false));
    try std.testing.expectError(error.DuplicateFeature, validate(&.{ valid, valid }, &.{source}, &.{"test-model"}, false));
    var missing = valid;
    missing.state = .missing;
    try std.testing.expectError(error.MissingCoverageLimitation, validate(&.{missing}, &.{source}, &.{"test-model"}, false));
    missing.limitation = "No implementation yet";
    missing.implementation = &.{};
    missing.checks = &.{};
    try std.testing.expectEqual(@as(usize, 1), try validate(&.{missing}, &.{source}, &.{}, false));
    missing.state = .present;
    try std.testing.expectError(error.UnbackedFeature, validate(&.{missing}, &.{source}, &.{}, false));
}

test "anchors require real Zig function bodies, not comments strings or prototypes" {
    const source =
        \\// fn removed() void {}
        \\const fake = "fn removed() void {}";
        \\extern fn declaration() void;
        \\const Model = struct { pub fn forward() void {} };
        \\test "real test" {}
    ;
    try std.testing.expect(try hasDeclaration(std.testing.allocator, source, .{ .path = "a.zig", .symbol = "forward" }));
    try std.testing.expect(!try hasDeclaration(std.testing.allocator, source, .{ .path = "a.zig", .symbol = "removed" }));
    try std.testing.expect(!try hasDeclaration(std.testing.allocator, source, .{ .path = "a.zig", .symbol = "declaration" }));
    try std.testing.expect(try hasDeclaration(std.testing.allocator, source, .{ .path = "a.zig", .symbol = "real test", .kind = .@"test" }));
    try std.testing.expect(!try hasDeclaration(std.testing.allocator, source, .{ .path = "a.zig", .symbol = "forward", .kind = .@"test" }));
    for ([_][]const u8{ "../outside.zig", "/absolute.zig", "a//b.zig", "a/./b.zig", "a\\b.zig" }) |path| try std.testing.expect(!localPath(path));
}
