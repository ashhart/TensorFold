//! Opt-in local lattice diagnostics; these records are never used as future-token inputs to inference.
const std = @import("std");
const frame = @import("cuda_draft_frame.zig");
const lp_words = 43;
const top = 20;

const WidePick = struct { id: u32, value: f32 };
fn better(a: WidePick, b: WidePick) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// A diagnostic heap retains the largest unary logits without changing the candidates served by the drafter.
fn widest(logits: []const u16, narrowed: bool, out: *[128]WidePick) !void {
    var count: usize = 0;
    for (logits, 0..) |bits, column| {
        const pick: WidePick = .{ .id = @intCast(if (narrowed and column >= 98304) column + 248032 - 98304 else column), .value = @bitCast(@as(u32, bits) << 16) };
        if (!std.math.isFinite(pick.value)) return error.NonfiniteDraftTrace;
        if (count < out.len) {
            var at = count;
            count += 1;
            out[at] = pick;
            while (at > 0) {
                const parent = (at - 1) / 2;
                if (!better(out[parent], out[at])) break;
                std.mem.swap(WidePick, &out[parent], &out[at]);
                at = parent;
            }
        } else if (better(pick, out[0])) {
            out[0] = pick;
            var at: usize = 0;
            while (at * 2 + 1 < count) {
                var child = at * 2 + 1;
                if (child + 1 < count and better(out[child], out[child + 1])) child += 1;
                if (!better(out[at], out[child])) break;
                std.mem.swap(WidePick, &out[at], &out[child]);
                at = child;
            }
        }
    }
    if (count != out.len) return error.BadDraftTrace;
    std.mem.sort(WidePick, out, {}, struct {
        fn before(_: void, a: WidePick, b: WidePick) bool {
            return better(a, b);
        }
    }.before);
}

/// Diagnostic-only readback of a completed head; no reference/future token IDs enter the forward or selection.
pub fn wide(d: anytype, items: anytype, logits: u64, vocab: usize, narrowed: bool) !void {
    const raw = (try @import("sys").environment.given("TF_DRAFT_WIDE_TRACE")) orelse return;
    if (!std.mem.eql(u8, raw, "1") or items.len == 0 or items[0].tokens.len == 0) return;
    if (items.len != 1 or d.c.topk != 16 or d.c.vocab != 248320 or d.e.head != .a4) return error.UnsupportedDraftTrace;
    const hold = items[0];
    const count = d.c.block - 1;
    const host = try d.gpa.alloc(u16, count * vocab);
    defer d.gpa.free(host);
    try d.e.d.check(d.e.d.api.cuMemcpyDtoH_v2(host.ptr, logits, host.len * 2), "cuMemcpyDtoH(wide draft trace)");
    const Row = struct { ids: [128]u32 = undefined, logit_bits: [128]u32 = undefined, hp_bits: []align(1) const u16 };
    const rows = try d.gpa.alloc(Row, count);
    defer d.gpa.free(rows);
    for (rows, 0..) |*row, i| {
        var picks: [128]WidePick = undefined;
        try widest(host[i * vocab ..][0..vocab], narrowed, &picks);
        row.* = .{ .hp_bits = std.mem.bytesAsSlice(u16, d.staged.bytes[frame.hp_at + i * d.c.rank * 2 ..][0 .. d.c.rank * 2]) };
        const words = std.mem.bytesAsSlice(u32, d.staged.bytes[frame.words_at + i * lp_words * 4 ..][0 .. lp_words * 4]);
        for (picks, 0..) |p, j| {
            row.ids[j] = p.id;
            row.logit_bits[j] = @bitCast(p.value);
            if (j < d.c.topk and (p.id != words[2 + j] or row.logit_bits[j] != words[2 + top + j])) return error.DraftTraceHeadMismatch;
        }
    }
    const value = .{
        .position = hold.position,
        .context_len = hold.tokens.len,
        .pending = hold.pending,
        .tail = hold.tokens[hold.tokens.len - @min(8, hold.tokens.len) ..],
        .block = d.c.block,
        .topk = 128,
        .rank = d.c.rank,
        .narrowed = narrowed,
        .rows = rows,
    };
    const bytes = try std.json.Stringify.valueAlloc(d.gpa, value, .{});
    defer d.gpa.free(bytes);
    std.debug.print("[DEBUG-dflash-wide] {s}\n", .{bytes});
}

/// Counterfactual target predictions are diagnostic proposals, never committed reference tokens.
pub fn verified(gpa: std.mem.Allocator, window: anytype, result: anytype) !void {
    const raw = (try @import("sys").environment.given("TF_DRAFT_FEEDBACK_TRACE")) orelse return;
    if (!std.mem.eql(u8, raw, "1")) return;
    const value = .{
        .position = window.positions[0],
        .pending = window.pending,
        .tokens = window.tokens,
        .parents = window.parents,
        .positions = window.positions,
        .draws = result.sampled,
        .sampled = window.stream.sampling != null,
    };
    const bytes = try std.json.Stringify.valueAlloc(gpa, value, .{});
    defer gpa.free(bytes);
    std.debug.print("[DEBUG-target-feedback] {s}\n", .{bytes});
}

pub fn emit(d: anytype, hold: anytype) !void {
    if (hold.tokens.len == 0) return;
    const raw = (try @import("sys").environment.given("TF_DRAFT_LATTICE_TRACE")) orelse return;
    if (!std.mem.eql(u8, raw, "1")) return;
    const Row = struct { ids: [top]u32 = @splat(0), logit_bits: [top]u32 = @splat(0), hp_bits: []align(1) const u16 };
    const rows = try d.gpa.alloc(Row, d.c.block - 1);
    defer d.gpa.free(rows);
    for (rows, 0..) |*row, i| {
        const at = d.selection_base + i;
        const words = std.mem.bytesAsSlice(u32, d.staged.bytes[frame.words_at + at * lp_words * 4 ..][0 .. lp_words * 4]);
        row.* = .{ .hp_bits = std.mem.bytesAsSlice(u16, d.staged.bytes[frame.hp_at + at * d.c.rank * 2 ..][0 .. d.c.rank * 2]) };
        @memcpy(row.ids[0..d.c.topk], words[2..][0..d.c.topk]);
        @memcpy(row.logit_bits[0..d.c.topk], words[2 + top ..][0..d.c.topk]);
    }
    const value = .{
        .position = hold.position,
        .context_len = hold.tokens.len,
        .pending = hold.pending,
        .lane = d.selection_base / (d.c.block - 1),
        .mask_keys = frame.segments(d.staged.bytes)[d.selection_base / (d.c.block - 1)].attention_rows,
        .tail = hold.tokens[hold.tokens.len - @min(8, hold.tokens.len) ..],
        .block = d.c.block,
        .topk = d.c.topk,
        .rank = d.c.rank,
        .sampled = hold.sampling != null,
        .rows = rows,
    };
    const bytes = try std.json.Stringify.valueAlloc(d.gpa, value, .{});
    defer d.gpa.free(bytes);
    std.debug.print("[DEBUG-dflash-lattice] {s}\n", .{bytes});
}
