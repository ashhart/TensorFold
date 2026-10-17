//! Physical two-slot prompt tap and drafted/plain equality gate on supplied target and drafter weights.
const std = @import("std");
const api = @import("engine_api");
const host = @import("qwen27_served");
const mtl = @import("metal");
const Box = struct {
    a: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    fn event(ptr: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (ev.*) {
            .tokens => |ids| b.tokens.appendSlice(b.a, ids) catch @panic("OOM"),
            .finished => |f| {
                b.done = f.reason;
            },
            else => {},
        }
    }
    fn wait(b: *Box) ![]const u32 {
        for (0..600_000) |_| {
            b.mutex.lockUncancelable(b.io);
            const done = b.done;
            b.mutex.unlock(b.io);
            if (done) |reason| {
                if (reason == .failed) return error.RequestFailed;
                return b.tokens.items;
            }
            try std.Io.sleep(b.io, .fromMilliseconds(10), .awake);
        }
        return error.Timeout;
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 7 or !std.mem.eql(u8, args[1], "--model") or
        !std.mem.eql(u8, args[3], "--draft") or !std.mem.eql(u8, args[5], "--tokens")) return error.Usage;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[6], a, .limited(16 << 20));
    const ids = (try std.json.parseFromSlice([]u32, a, text, .{})).value;
    if (ids.len < 2080) return error.PromptTooShort;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var why: []const u8 = "";
    const h = try host.open(init.gpa, io, args[2], .{ .context = 8192, .streams = 2, .streams_fixed = true, .drafter = args[4], .drafter_bits = 4, .cache_gib = 0 }, a, &why);
    defer host.close(h);
    h.host.cache = null;
    h.back.release_hook = null;
    var next: api.Id = 1;
    for ([_]usize{ 128, 129, 256, 512, 513, 2048, 2051 }) |length| {
        var expected: [2][]const u32 = undefined;
        for (0..2) |i| {
            var box = Box{ .a = a, .io = io };
            const request = api.Request{ .prompt = ids[i * 17 ..][0..length], .max_tokens = 40, .drafts = false };
            try h.engine().submit(next, &request, .{ .ctx = &box, .event = Box.event });
            next += 1;
            expected[i] = try box.wait();
        }
        for ([_]bool{ false, true }) |drafting| {
            var boxes: [2]Box = @splat(.{ .a = a, .io = io });
            var requests: [2]api.Request = undefined;
            for (&requests, &boxes, 0..) |*request, *box, i| {
                request.* = .{ .prompt = ids[i * 17 ..][0..length], .max_tokens = 40, .drafts = drafting };
                try h.engine().submit(next, request, .{ .ctx = box, .event = Box.event });
                next += 1;
            }
            for (&boxes, expected) |*box, want| {
                if (!std.mem.eql(u32, want, try box.wait())) return error.ConcurrentTokensDiffer;
            }
        }
        std.debug.print("PASS {d} rows: two slots drafted/plain and concurrent/solo token IDs\n", .{length});
    }
}
