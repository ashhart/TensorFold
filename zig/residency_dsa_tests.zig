//! The paired GLM inventory uses actual prompt and decode scratch constructors without a transport or GPU.
const std = @import("std");
const mtl = @import("metal");
const g = @import("src/families/glm_dsa/glm_dsa.zig");
const a = std.testing.allocator;
const Buffers = struct {
    device: mtl.Device,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    fn take(b: *@This(), bytes: usize) !mtl.Buffer {
        const buffer = try b.device.buffer(bytes, 0);
        errdefer buffer.deinit();
        try b.owned.append(a, buffer);
        return buffer;
    }
    fn fill(b: *@This(), owner: anytype) !void {
        const info = @typeInfo(@TypeOf(owner.*)).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Buffer) @field(owner.*, name) = try b.take(64);
            if (T == ?mtl.Buffer) @field(owner.*, name) = try b.take(64);
        }
    }
    fn deinit(b: *@This()) void {
        for (b.owned.items) |buffer| buffer.deinit();
        b.owned.deinit(a);
    }
};

test "residency owner GLM DSA: actual scratch, packed weights, transport, MTP and snapshots share the native inventory" {
    for ([_]bool{ false, true }) |mtp| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            var buffers: Buffers = .{ .device = record.device() };
            defer buffers.deinit();
            var e: g.engine.Engine = undefined;
            e.gpa = a;
            e.device = record.device();
            e.queue = record.queue();
            e.context = 4;
            e.resident = null;
            e.m.c = .{ .hidden = 64, .layers = 1, .vocab = 32, .experts = 4, .topk = 1, .heads = 2, .dense_inter = 128, .moe_inter = 128, .q_lora = 64, .kv_lora = 64, .i_heads = 1, .i_topk = 4 };
            try buffers.fill(&e.m);
            try buffers.fill(&e.m.K);
            try buffers.fill(&e.m.vocab.k);
            try buffers.fill(&e.m.head);
            var layer: [1]g.layer.Layer = undefined;
            try buffers.fill(&layer[0].k);
            layer[0].cache = try buffers.take(4 * 1152);
            layer[0].index_cache = try buffers.take(4 * 256);
            layer[0].desc = try buffers.take(64);
            e.m.layers = &layer;
            e.m.mtp = null;
            if (mtp) {
                var extra: g.layer.Layer = undefined;
                try buffers.fill(&extra.k);
                extra.cache = try buffers.take(4 * 1152);
                extra.index_cache = try buffers.take(4 * 256);
                extra.desc = null;
                e.m.mtp = extra;
            }
            var pair: @import("src/core/pair.zig").Pair = undefined;
            pair.wbuf = try buffers.take(64);
            e.pair = &pair;
            e.s = try g.gpu.Scratch.init(e.device, &e.m.c, e.context);
            defer e.s.deinit();
            e.q = try g.gpu.Prompt.init(e.device, &e.m.c, e.context);
            defer e.q.deinit();
            for (&e.pass) |*p| try buffers.fill(p);
            e.xs = try buffers.take((2 * g.gpu.max_rows + 1) * e.m.c.hidden * 4);
            var slots = [_]g.engine.Slot{.{ .used = true, .len = 1 }};
            e.slots = &slots;
            e.snaps = .empty;
            defer e.snaps.deinit(a);
            try g.residency.hold(&e, std.testing.io, null);
            defer g.residency.drop(&e);
            try record.checkInitial();
            try record.check();
            var warm = g.residency.target(&e);
            mtl.keepalive.Target.tick(&warm);
            try std.testing.expect(record.copies > 0 and record.uses > 0);
            record.fail_at = record.allocations.items.len;
            try std.testing.expectError(error.InjectedBufferFailure, g.snap.save(&e, 1, 0, 1));
            record.fail_at = null;
            try record.check();
            var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 1 });
            e.gpa = failing.allocator();
            const refused = g.snap.save(&e, 1, 0, 1);
            e.gpa = a;
            try std.testing.expectError(error.OutOfMemory, refused);
            try record.check();
            _ = try g.snap.save(&e, 1, 0, 1);
            defer g.snap.drop(&e, 1);
            e.resident.?.held.flush();
            try record.check();
            g.residency.drop(&e);
            record.refuse_set = true;
            try g.residency.hold(&e, std.testing.io, null);
            try std.testing.expect(e.resident == null and g.residency.target(&e).scratch == null);
            try record.check();
        }
        try record.allFreed();
    }
}
