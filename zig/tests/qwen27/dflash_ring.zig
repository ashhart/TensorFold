//! Full draft-head logits over identical synthetic KV, with and without the wrapped gather.
const std = @import("std");
const q = @import("tensorfold").qwen27;
const df = q.dflash;
const Model = df.runtime_model.Model;

const Readback = struct {
    draft: *Model,
    words: []u16,
    compare: bool = false,

    fn logits(ptr: *anyopaque, _: u64, output: df.execution.Output, _: df.config.Config) !void {
        const self: *Readback = @ptrCast(@alignCast(ptr));
        const view = output.head.logits;
        const frame = &self.draft.backend.frame;
        const ref = try frame.ref(view);
        try frame.finish();
        if (@as(usize, view.rows) * view.width != self.words.len) return error.DraftLogitShape;
        const data: [*]const u16 = @ptrCast(@alignCast(ref.buf.contents() + ref.off));
        for (0..view.rows) |row| {
            const actual = data[row * view.stride ..][0..view.width];
            const expected = self.words[row * view.width ..][0..view.width];
            if (self.compare) {
                if (!std.mem.eql(u16, expected, actual)) return error.DraftRingLogitBytes;
            } else {
                @memcpy(expected, actual);
            }
        }
    }
};

pub fn run(a: std.mem.Allocator, target: *q.model.Model) !void {
    const window = 2048;
    const draft = try @import("dflash_graph.zig").createWindow(a, target, window);
    defer draft.deinit();
    const original = draft.backend.ring_read;
    defer draft.backend.ring_read = original;
    const words = try a.alloc(u16, 15 * draft.graph.config.vocab);
    defer a.free(words);
    var readback = Readback{ .draft = draft, .words = words };
    var random: std.Random.DefaultPrng = .init(0x3900);
    for (draft.backend.caches) |cache| {
        for ([_]@import("metal").Buffer{ cache.keys, cache.values }) |buffer| {
            for (buffer.slice(u16, window * draft.graph.config.kvWidth())) |*item| {
                const f = @as(f32, @floatFromInt(random.random().intRangeLessThan(i32, -128, 129))) / 256;
                item.* = @truncate(@as(u32, @bitCast(f)) >> 16);
            }
        }
    }
    for ([_]u64{ 2047, 2048, 2049, 4096, 9000 }) |length| {
        const context = df.operators.Context{ .cache = 1, .begin = length - (window - 1), .end = length };
        draft.backend.context = context;
        for ([_]bool{ false, true }) |enabled| {
            draft.backend.ring_read = enabled;
            readback.compare = enabled;
            try df.execution.forwardToBlock(draft.backend.ops(), &draft.graph, draft.preparation, context, 7, 16, .{
                .ptr = &readback,
                .call = Readback.logits,
            });
        }
        std.debug.print("DFlash2 ring L={d}: all {d} draft logits byte-equal\n", .{ length, words.len });
    }
}
