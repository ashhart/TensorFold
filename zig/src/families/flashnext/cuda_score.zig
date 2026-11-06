//! Decisions' label scores on CUDA: the prompt through a fresh cache, then its last row's label logits and logsumexp.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");

/// One logit per label at the row after `ids` (bf16 widened), and the row's logsumexp in f64 when `mass` (else 0).
pub fn labels(comptime Tri: type, e: *prompt.Engine, tri: Tri, context: usize, ids: []const u32, wanted: []const u32, logits: []f64, mass: bool) !f64 {
    // every refusal before a cache is made
    if (ids.len == 0 or ids.len > context or ids.len > prompt.max_tokens) return error.PromptOutsideContext;
    if (wanted.len == 0 or wanted.len != logits.len) return error.UnexpectedTensor;
    for (wanted) |t| if (t >= prompt.vocabulary) return error.LabelOutsideVocabulary;
    var seq = try e.newSeq(ids.len);
    defer seq.deinit(e.gpa);
    try e.forward(Tri, tri, &seq, ids, .{});
    const row = std.mem.bytesAsSlice(u16, seq.logits[0 .. prompt.vocabulary * 2]);
    for (wanted, logits) |t, *v| {
        v.* = mtp.promote(std.mem.littleToNative(u16, row[t]));
        if (!std.math.isFinite(v.*)) return error.NonFiniteScore;
    }
    return if (mass) logsumexp(row) else 0;
}

/// The bf16 row's logsumexp in f64; a non-finite row is refused.
fn logsumexp(row: []align(1) const u16) !f64 {
    var top: f64 = -std.math.inf(f64);
    for (row) |b| {
        const v: f64 = mtp.promote(std.mem.littleToNative(u16, b));
        if (!std.math.isFinite(v)) return error.NonFiniteScore;
        top = @max(top, v);
    }
    var sum: f64 = 0;
    for (row) |b| sum += @exp(@as(f64, mtp.promote(std.mem.littleToNative(u16, b))) - top);
    return top + @log(sum);
}

test "the logsumexp of a bf16 row in f64, and a non-finite row refused" {
    const one: u16 = 0x3f80; // 1.0
    const two: u16 = 0x4000; // 2.0
    const row = [_]u16{ one, two, one };
    const want = 2.0 + @log(1.0 + 2.0 * @exp(@as(f64, -1.0)));
    try std.testing.expectApproxEqRel(want, try logsumexp(&row), 1e-15);
    try std.testing.expectError(error.NonFiniteScore, logsumexp(&[_]u16{ one, 0x7f80 }));
}

pub fn CapabilityCases(comptime Cuda: type) type {
    return struct {
        test "decision capability: one rank scores and two ranks refuse before a score runs" {
            var self: Cuda = undefined;
            self.head = null;
            self.ranks = null;
            try std.testing.expect(self.backend().vtable.score != null);
            self.ranks = @ptrFromInt(16);
            try std.testing.expect(self.backend().vtable.score == null);
        }

        test "decision capability: the ranked table outlives backend(), as the scoring table does" {
            var self: Cuda = undefined;
            self.head = null;
            for ([_]usize{ 0, 16 }) |r| {
                self.ranks = if (r == 0) null else @ptrFromInt(r);
                // a real call: inlined, a table built in backend() would sit in this frame and survive clobber()
                const vt = @call(.never_inline, Cuda.backend, .{&self}).vtable;
                var here: u8 = 0;
                std.mem.doNotOptimizeAway(&here);
                const apart = @max(@intFromPtr(vt), @intFromPtr(&here)) - @min(@intFromPtr(vt), @intFromPtr(&here));
                try std.testing.expect(apart > 1 << 20); // static data, not this thread's stack
                clobber();
                // read before backend() runs again: a table in its frame would be rebuilt at the same address
                const step = vt.prefill_step;
                const scores = vt.score != null;
                try std.testing.expect(step != null and step == self.backend().vtable.prefill_step);
                try std.testing.expectEqual(r == 0, scores);
            }
        }

        /// Writes over the stack below the caller, where a table built inside backend() would have lived.
        noinline fn clobber() void {
            var junk: [4096]u8 = undefined;
            @memset(&junk, 0xa5);
            std.mem.doNotOptimizeAway(&junk);
        }
    };
}

/// The native adapter's one-rank score hook, kept with its allocation and numerical implementation.
pub fn Hook(comptime Cuda: type, comptime Tri: type) type {
    return struct {
        pub fn call(ptr: *anyopaque, ids: []const u32, wanted: []const u32, logits: []f64, mass: bool) anyerror!f64 {
            const self: *Cuda = @ptrCast(@alignCast(ptr));
            if (self.ranks != null) return error.UnsupportedRankOperation;
            return labels(Tri, self.engine, self.tri(), self.context, ids, wanted, logits, mass);
        }
    };
}

/// A ranked backend keeps its existing callbacks and deliberately advertises no scoring capability.
pub fn withoutScore(comptime scored: @import("lanes").backend.Backend.VTable) @import("lanes").backend.Backend.VTable {
    var v = scored;
    v.score = null;
    return v;
}
