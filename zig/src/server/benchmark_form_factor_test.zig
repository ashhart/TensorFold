//! Coarse hardware labels survive receipts and manifests without exporting product identifiers.
const std = @import("std");
const json = @import("json");
const form = @import("benchmark_form_factor.zig");
const receipt = @import("benchmark_receipt.zig");
const check = @import("benchmark_check.zig");
const manifest = @import("benchmark_manifest.zig");
const host = @import("benchmark_host.zig");
const golden = @embedFile("fixtures/benchmark_golden.json");
const legacy = @embedFile("fixtures/benchmark_v1.json");
const now_ms: i64 = 1767225600000 + 60_000;

test "form factor: exact product lookup distinguishes the six families and never guesses unknown IDs" {
    const cases = [_]struct { []const u8, form.FormFactor }{
        .{ "Mac17,16", .mac_mini },   .{ "Macmini9,1", .mac_mini },        .{ "Mac18,5", .mac_mini },
        .{ "Mac17,14", .mac_studio }, .{ "Mac15,14", .mac_studio },        .{ "MacBookPro18,3", .macbook_pro },
        .{ "Mac16,6", .macbook_pro }, .{ "MacBookAir10,1", .macbook_air }, .{ "Mac17,3", .macbook_air },
        .{ "iMac21,1", .imac },       .{ "Mac16,3", .imac },               .{ "Mac14,8", .mac_pro },
        .{ "Mac17,2", .macbook_pro }, .{ "Mac17,8", .macbook_pro },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c[1], form.fromProduct(c[0]).?);
        try std.testing.expectEqual(c[1], form.FormFactor.fromLabel(c[1].label()).?);
    }
    for ([_][]const u8{ "", "Mac99,99", "Mac17,160", "Macmini99,1", "MacBookPro99,1", "Mac17,16\n", "mac17,16", " Mac17,16", "Mac17,16\x00extra" }) |s|
        try std.testing.expect(form.fromProduct(s) == null);
}

test "form factor: only the six coarse labels enter receipts and manifests" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ golden, legacy }) |bytes| {
        for (std.enums.values(form.FormFactor)) |family| {
            const v = (try json.parse(a, bytes)).ok;
            try v.get("hardware").?.object.put(a, "form_factor", .{ .string = family.label() });
            var why: []const u8 = "";
            try std.testing.expect(check.receipt(v, .save, now_ms, &why));
            try std.testing.expect(check.receipt(v, .publish, now_ms, &why));
            const encoded = try receipt.canonical(a, v);
            const decoded = (try json.parse(a, encoded)).ok;
            try std.testing.expect(check.receipt(decoded, .save, now_ms, &why));
            try std.testing.expectEqualStrings(encoded, try receipt.canonical(a, decoded));
            const facts = (try manifest.parse(a, try manifestBody(a, v), "cuda")).?;
            try std.testing.expectEqual(family, facts.hardware.form_factor.?);
        }
    }
}

test "form factor: absent and null keep old receipt bytes and manifest acceptance" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ golden, legacy }) |bytes| {
        const old = (try json.parse(a, bytes)).ok;
        var why: []const u8 = "";
        try std.testing.expect(check.receipt(old, .save, now_ms, &why));
        try std.testing.expect(check.receipt(old, .publish, now_ms, &why));
        try std.testing.expectEqualStrings(std.mem.trimEnd(u8, bytes, "\n"), try receipt.canonical(a, old));
        try std.testing.expect((try manifest.parse(a, try manifestBody(a, old), "cuda")).?.hardware.form_factor == null);
        try old.get("hardware").?.object.put(a, "form_factor", .null);
        try std.testing.expect(check.receipt(old, .publish, now_ms, &why));
        const encoded = try receipt.canonical(a, old);
        try std.testing.expectEqualStrings(encoded, try receipt.canonical(a, (try json.parse(a, encoded)).ok));
        try std.testing.expect((try manifest.parse(a, try manifestBody(a, old), "cuda")).?.hardware.form_factor == null);
    }
}

test "form factor: arbitrary text, product IDs, wrong JSON types and extra hardware keys are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "\"Mac17,16\"", "\"Mac mini M5 Pro\"", "\"mac mini\"", "\"Mac mini\\n\"", "\"\"", "true", "7", "[]", "{}" }) |text| {
        const v = (try json.parse(a, golden)).ok;
        try v.get("hardware").?.object.put(a, "form_factor", (try json.parse(a, text)).ok);
        for ([_]check.Mode{ .save, .publish }) |mode| {
            var why: []const u8 = "";
            try std.testing.expect(!check.receipt(v, mode, now_ms, &why));
            try std.testing.expectEqualStrings("hardware.form_factor must be a known Mac family or null", why);
        }
        try std.testing.expect((try manifest.parse(a, try manifestBody(a, v), "cuda")) == null);
    }
    for ([_]bool{ false, true }) |has_family| {
        const v = (try json.parse(a, golden)).ok;
        if (has_family) try v.get("hardware").?.object.put(a, "form_factor", .{ .string = "Mac mini" });
        try v.get("hardware").?.object.put(a, "model_identifier", .{ .string = "Mac17,16" });
        var why: []const u8 = "";
        try std.testing.expect(!check.receipt(v, .save, now_ms, &why));
        try std.testing.expectEqualStrings("hardware must hold exactly its fields", why);
    }
}

test "form factor: the writer emits a coarse family only and leaves non-Mac probes unknown" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const known = try built(a, host.mac(a, "Apple M5 Pro", 48 << 30, "", "Mac17,16").hardware);
    try std.testing.expectEqualStrings("Mac mini", known.get("hardware").?.strField("form_factor").?);
    const encoded = try receipt.canonical(a, known);
    for ([_][]const u8{ "Mac17,16", "hw.product", "model_identifier" }) |s|
        try std.testing.expect(std.mem.indexOf(u8, encoded, s) == null);
    const unknown = try built(a, host.mac(a, "Apple M5 Pro", 48 << 30, "", "Mac99,99").hardware);
    try std.testing.expect(unknown.get("hardware").?.get("form_factor") == null);
    try std.testing.expect(host.mac(a, null, null, "", null).hardware.form_factor == null);
    try std.testing.expect(host.linux(a, "ARM Cortex", 1000, "", null).hardware.form_factor == null);
    try std.testing.expect(host.rocm(a, "AMD EPYC", 1000, "", null).hardware.form_factor == null);
}

fn manifestBody(a: std.mem.Allocator, v: json.Value) ![]const u8 {
    const root = try json.newObject(a);
    try root.put(a, "schema_version", try json.intValue(a, @as(u8, 1)));
    try root.put(a, "runtime", v.get("runtime").?);
    try root.put(a, "model", .{ .object = try json.newObject(a) });
    const hardware = (try json.parse(a, try json.stringify(a, v.get("hardware").?, .{}))).ok;
    try hardware.object.put(a, "source", .{ .string = "partial" });
    try root.put(a, "hardware", hardware);
    return json.stringify(a, .{ .object = root }, .{});
}

fn built(a: std.mem.Allocator, hardware: receipt.Hardware) !json.Value {
    return receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", .{ .local = false }, .{ .tensorfold_version = "1.2.1", .backend = "mlx", .python_version = "none", .platform = "macos" }, hardware, .{ .tokens = 4, .repetitions = 1, .temperatures = &.{0}, .serial = false, .context_tokens = null, .managed = false }, &.{});
}
