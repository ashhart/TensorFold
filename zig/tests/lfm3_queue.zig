//! The real one-token backend queue against serial forward and the unchanged host draw.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.lfm3;
const lanes = tf.lanes;

pub fn check(m: *q.Model, ids: []const u32) !void {
    const a = m.gpa;
    const owner = blk: {
        const out = try a.create(q.backend.Metal);
        errdefer a.destroy(out);
        out.* = .{ .gpa = a, .model = m, .options = .{ .capacity = 512 }, .scratch = try q.state.Scratch.init(a, m.device, &m.config.layer_types, 128) };
        break :blk out;
    };
    defer owner.deinit();
    const backend = owner.backend();
    const modes = [_]?lanes.Sampling{ null, .{ .seed = 7, .temperature = 0 }, .{ .seed = 11, .temperature = 1, .top_k = 16 } };
    var solo = try q.state.Scratch.init(a, m.device, &m.config.layer_types, 1);
    defer solo.deinit();
    for (modes) |mode| {
        var stream = try lanes.Stream.init(a, .{ .id = "queue-exact", .prompt = ids, .max_new = 16, .sampling = mode });
        defer stream.deinit(a);
        try backend.prefill(&stream);
        defer backend.release(&stream);
        const actual = owner.caches.get(&stream).?;
        var serial = try q.state.Cache.init(a, m.device, &m.config.layer_types, actual.capacity);
        defer serial.deinit();
        for (ids) |id| try q.forward.run(m, &solo, &.{.{ .cache = &serial, .rows = 1 }}, &.{id}, false, .all);
        try @import("lfm3_exact.zig").equal(&serial, actual);
        var token = try q.backend.draw(a, serial.logits.slice(u16, q.config.vocab), mode, ids.len, &.{});
        for (0..8) |i| {
            const position = ids.len + i + 1;
            const handle = try backend.queue(&stream, .{ .value = token }, position);
            const got = try backend.read(handle);
            try q.forward.run(m, &solo, &.{.{ .cache = &serial, .rows = 1 }}, &.{token}, false, .all);
            const want = try q.backend.draw(a, serial.logits.slice(u16, q.config.vocab), mode, position, &.{});
            if (got != want) return error.QueuePickMismatch;
            try @import("lfm3_exact.zig").equal(&serial, actual);
            token = want;
        }
    }
}
