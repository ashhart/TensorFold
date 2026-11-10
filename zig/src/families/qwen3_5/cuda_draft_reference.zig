//! One opt-in eager DFlash frame and its real context, for independent model arithmetic checks.
const std = @import("std");

fn directory() !?[]const u8 {
    return @import("sys").environment.given("TF_DRAFT_REFERENCE_DIR");
}

fn write(d: anytype, name: []const u8, bytes: []const u8) !void {
    const dir = (try directory()) orelse return;
    return writeTo(d, dir, name, bytes);
}

fn writeTo(d: anytype, dir: []const u8, name: []const u8, bytes: []const u8) !void {
    const path = try std.fs.path.join(d.gpa, &.{ dir, name });
    defer d.gpa.free(path);
    if (std.fs.path.dirname(path)) |parent| try std.Io.Dir.cwd().createDirPath(d.io, parent);
    try std.Io.Dir.cwd().writeFile(d.io, .{ .sub_path = path, .data = bytes });
}

fn buffer(d: anytype, name: []const u8, device: u64, rows: usize, width: usize) !void {
    const dir = (try directory()) orelse return;
    return bufferTo(d, dir, name, device, rows, width);
}

fn bufferTo(d: anytype, dir: []const u8, name: []const u8, device: u64, rows: usize, width: usize) !void {
    const bytes = try std.math.mul(usize, try std.math.mul(usize, rows, width), 2);
    const host = try d.gpa.alloc(u8, bytes);
    defer d.gpa.free(host);
    try d.e.stream.synchronize();
    try d.e.d.check(d.e.d.api.cuMemcpyDtoH_v2(host.ptr, device, bytes), "cuMemcpyDtoH(DFlash reference)");
    try writeTo(d, dir, name, host);
}

// Private single-stream training capture; the serving path never reads these files or reference continuations.
var dataset_started = false;
var dataset_case: usize = 0;
var dataset_end: u64 = 0;

fn datasetDirectory() !?[]const u8 {
    return @import("sys").environment.given("TF_DRAFT_DATASET_DIR");
}

fn datasetContext(d: anytype, start: u64, rows: usize) !void {
    const dir = (try datasetDirectory()) orelse return;
    if (d.streams.count() == 0) return; // synthetic startup contexts
    if (d.streams.count() != 1) return error.SingleStreamDatasetRequired;
    const norm = (try @import("sys").environment.given("TF_DRAFT_QWEN3_NORM")) orelse return error.CorrectedNormalizationDatasetRequired;
    if (!std.mem.eql(u8, norm, "1")) return error.CorrectedNormalizationDatasetRequired;
    if (start == 0) {
        if (dataset_started) dataset_case += 1;
        dataset_started = true;
        dataset_end = 0;
    }
    if (!dataset_started or start != dataset_end) return error.NoncontiguousDatasetContext;
    var storage: [128]u8 = undefined;
    for ([_][]const u8{ "taps", "fc", "context" }, [_]u64{ d.taps_in, d.th, d.thn }, 0..) |name, ptr, i| {
        const file = try std.fmt.bufPrint(&storage, "case-{d}/context-{d}-{d}-{s}.bf16", .{ dataset_case, start, rows, name });
        try bufferTo(d, dir, file, ptr, rows, if (i == 0) d.c.hidden * d.c.n_taps else d.c.hidden);
    }
    dataset_end = try std.math.add(u64, start, rows);
}

fn datasetFrame(d: anytype, items: anytype, contexts: anytype) !void {
    const dir = (try datasetDirectory()) orelse return;
    if (items.len == 0 or items[0].tokens.len == 0) return;
    if (items.len != 1 or contexts.len != 1 or d.streams.count() != 1 or !dataset_started) return error.SingleStreamDatasetRequired;
    if (contexts[0].end != dataset_end) return error.NoncontiguousDatasetContext;
    const value = .{ .config = d.c, .case = dataset_case, .begin = contexts[0].begin, .end = contexts[0].end, .position = items[0].position, .pending = items[0].pending, .known_tokens = items[0].tokens, .normalization = "Qwen3 BF16 row and gain rounding", .diagnostic_only = true };
    const json = try std.json.Stringify.valueAlloc(d.gpa, value, .{});
    defer d.gpa.free(json);
    var storage: [128]u8 = undefined;
    const file = try std.fmt.bufPrint(&storage, "case-{d}/frame-{d}.json", .{ dataset_case, items[0].position });
    try writeTo(d, dir, file, json);
}

/// Prompt context is copied after its existing FC and normalization, before any drafter layer consumes it.
pub fn absorbed(d: anytype, start: u64, rows: usize) !void {
    try datasetContext(d, start, rows);
    if ((try directory()) == null) return;
    if (d.reference_done or d.streams.count() == 0) return; // reserve's synthetic warmups are not a reference
    if (d.streams.count() != 1 or d.graphs != null) return error.EagerSingleStreamReferenceRequired;
    var storage: [96]u8 = undefined;
    const names = [_][]const u8{ "taps", "fc", "context" };
    const pointers = [_]u64{ d.taps_in, d.th, d.thn };
    for (names, pointers, 0..) |name, ptr, i| {
        const file = try std.fmt.bufPrint(&storage, "context-{d}-{d}-{s}.bf16", .{ start, rows, name });
        try buffer(d, file, ptr, rows, if (i == 0) d.c.hidden * d.c.n_taps else d.c.hidden);
    }
}

pub fn begin(d: anytype, items: anytype, contexts: anytype) !bool {
    try datasetFrame(d, items, contexts);
    if ((try directory()) == null) return false;
    if (items.len == 0 or items[0].tokens.len == 0 or d.reference_done) return false;
    if (items.len != 1 or contexts.len != 1 or d.graphs != null or d.e.head != .a4) return error.EagerSingleStreamReferenceRequired;
    d.reference_capture = true;
    const value = .{
        .config = d.c,
        .begin = contexts[0].begin,
        .end = contexts[0].end,
        .pending = items[0].pending,
        .position = items[0].position,
        .known_tokens = items[0].tokens,
        .bf16_mask = (try @import("sys").environment.given("TF_DRAFT_BF16")) orelse "0",
        .precision = "BF16 boundary tensors; target checkpoint and verification unchanged",
    };
    const json = try std.json.Stringify.valueAlloc(d.gpa, value, .{});
    defer d.gpa.free(json);
    try write(d, "frame.json", json);
    return true;
}

pub fn point(d: anytype, name: []const u8, device: u64, rows: usize, width: usize) !void {
    if ((try directory()) == null) return;
    if (!d.reference_capture) return;
    var storage: [96]u8 = undefined;
    const file = try std.fmt.bufPrint(&storage, "{s}-{d}-{d}.bf16", .{ name, rows, width });
    try buffer(d, file, device, rows, width);
}

pub fn layer(d: anytype, index: usize, name: []const u8, device: u64, rows: usize, width: usize) !void {
    if ((try directory()) == null) return;
    if (!d.reference_capture) return;
    var storage: [96]u8 = undefined;
    const file = try std.fmt.bufPrint(&storage, "layer{d}-{s}-{d}-{d}.bf16", .{ index, name, rows, width });
    try buffer(d, file, device, rows, width);
}

pub fn complete(d: anytype) !void {
    if ((try directory()) == null) return;
    if (!d.reference_capture) return;
    try write(d, "COMPLETE", "DFlash reference boundary capture completed\n");
    d.reference_done = true;
    d.reference_capture = false;
}
