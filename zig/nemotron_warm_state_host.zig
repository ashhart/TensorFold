//! Exercise Nemotron's actual cache slots and snapshot copies with host command buffers, without shader arithmetic.
const std = @import("std");
const lanes = @import("lanes");
const mtl = @import("metal");
const st = @import("src/families/nemotron/state.zig");
const snap = @import("src/families/nemotron/snapshot.zig");
const pass = @import("src/families/nemotron/prefill_steps.zig");
const flight = @import("src/families/nemotron/prompt_flight.zig");
const Config = @import("src/families/nemotron/config.zig").Config;
const t = std.testing;
const a = t.allocator;
const count = 16385;
const boundary = 8192;
const Kind = enum(u2) { prefill, step, verify, draft };

fn config() Config {
    var c: Config = undefined;
    c.layers = 2;
    c.kinds[0..2].* = .{ .mamba, .attention };
    c.conv_kernel = 3;
    c.mamba_heads = 1;
    c.mamba_head_dim = 2;
    c.groups = 1;
    c.state = 2;
    c.head_dim = 2;
    c.kv_heads = 2;
    return c;
}
const Kernels = struct {
    pub fn get(_: Kernels, name: []const u8) u8 {
        std.debug.assert(std.mem.eql(u8, name, "tf_copy_u32"));
        return 0;
    }
};
const Model = struct { config: Config, device: mtl.Device = .{}, kernels: Kernels = .{} };
const Copy = struct { src: []const u8, dst: []u8 };
const Encoder = struct {
    buffers: [2][]u8 = undefined,
    copies: [32]Copy = undefined,
    n: usize = 0,
    pub fn alongside(_: *Encoder) void {}
    pub fn pipe(_: *Encoder, _: u8) void {}
    pub fn buf(e: *Encoder, b: mtl.Buffer, off: usize, i: usize) void {
        e.buffers[i] = b.data[off..];
    }
    pub fn run(e: *Encoder, grid: [3]usize, _: [3]usize) void {
        const n = grid[0] * 4;
        e.copies[e.n] = .{ .src = e.buffers[0][0..n], .dst = e.buffers[1][0..n] };
        e.n += 1;
    }
};
const Body = struct {
    b: *Backend,
    s: *lanes.Stream,
    at: usize,
    rows: usize,
    prior: u32 = 0,
    slot: u32 = 0,
    fn run(j: Body) void {
        const b = j.b;
        const c = &b.cache.?;
        const conv_n = (b.m.config.conv_kernel - 1) * b.m.config.convDim() * 2 / 4;
        const ssm_n = b.m.config.mamba_heads * b.m.config.mamba_head_dim * b.m.config.state;
        const conv = b.pool.conv[0].slice(u32, b.pool.slots * conv_n);
        const ssm = b.pool.ssm[0].slice(u32, b.pool.slots * ssm_n);
        var value = ssm[j.prior * ssm_n];
        for (j.at..j.at + j.rows) |row| {
            value = value *% 16777619 ^ j.s.prompt()[row];
            writeKv(c.kv[0], row, value);
            if (c.mtp) |head| if (row + 1 < j.s.prompt_len) writeKv(head, row, value ^ j.s.prompt()[row + 1]);
        }
        for (conv[j.slot * conv_n ..][0..conv_n], 0..) |*v, i| v.* = value +% @as(u32, @intCast(i));
        for (ssm[j.slot * ssm_n ..][0..ssm_n], 0..) |*v, i| v.* = value ^ @as(u32, @intCast(i));
        b.completed = j.at + j.rows;
    }
};
fn writeKv(kv: st.Kv, row: usize, value: u32) void {
    for (0..2) |h| {
        kv.k.slice(u32, 2 * kv.capacity)[h * kv.capacity + row] = value +% @as(u32, @intCast(h));
        kv.v.slice(u32, 2 * kv.capacity)[h * kv.capacity + row] = ~value ^ @as(u32, @intCast(h));
    }
}
const Command = struct {
    body: ?Body = null,
    encoder: Encoder = .{},
    pub fn wait(c: Command) void {
        if (c.body) |j| j.run();
        for (c.encoder.copies[0..c.encoder.n]) |copy| @memcpy(copy.dst, copy.src);
    }
    pub fn release(_: Command) void {}
    pub fn failure(_: Command) ?[]const u8 {
        return null;
    }
    pub fn gpuSeconds(_: Command) f64 {
        return 0;
    }
};
const Flight = struct { end: u64, cb: Command, kind: Kind, prompt: ?flight.Prompt = null };
const Backend = struct {
    gpa: std.mem.Allocator = a,
    m: *Model,
    pool: st.Pool,
    cache: ?st.Cache = null,
    with_head: bool,
    residency: ?mtl.ResidencySet.Follow = null,
    next: u64 = 1,
    landed: u64 = 0,
    last_ms: f64 = 0,
    gpu_ms: f64 = 0,
    timing: struct { ms: [4]f64 = @splat(0), count: [4]usize = @splat(0) } = .{},
    flights: std.ArrayList(Flight) = .empty,
    completed: usize = 0,
    first: ?usize = null,
    processed: usize = 0,
    pub fn submit(b: *Backend, kind: Kind, end: u64, body: anytype) !void {
        var command: Command = .{};
        if (@TypeOf(body) == Body) {
            var j = body;
            j.prior = b.cache.?.slot;
            j.slot = try b.pool.take();
            b.cache.?.advance(&b.pool, j.rows, j.slot);
            if (b.with_head) b.cache.?.mtp_len += pass.absorbed(j.at, j.rows, j.s.prompt_len);
            if (b.first == null) b.first = j.at;
            b.processed += j.rows;
            command.body = j;
        } else try body.encode(b, &command.encoder);
        try b.flights.append(a, .{ .end = end, .kind = kind, .cb = command });
    }
    pub fn land(b: *Backend) !void {
        try flight.land(b);
    }
    pub fn drain(b: *Backend) !void {
        while (b.flights.items.len > 0) try b.land();
    }
    fn release(b: *Backend) !void {
        try b.drain();
        if (b.cache) |*c| c.deinit(&b.pool);
        b.cache = null;
        b.completed = 0;
        b.first = null;
        b.processed = 0;
    }
    fn deinit(b: *Backend) void {
        b.release() catch unreachable;
        b.pool.deinit();
        b.flights.deinit(a);
    }
};
const Ops = struct {
    b: *Backend,
    pub fn cap(_: Ops) usize {
        return boundary;
    }
    pub fn decision(_: Ops, _: *lanes.Stream) ?usize {
        return null;
    }
    pub fn open(o: Ops, s: *lanes.Stream) !usize {
        try flight.checkpoint(s);
        try o.b.drain();
        o.b.cache = try st.Cache.init(o.b.m.device, o.b.m.config, count + 1, o.b.with_head);
        s.cached = 0;
        if (s.reuse.saved) |saved| {
            const kept: *snap.Snap = @ptrCast(@alignCast(saved));
            try snap.restore(o.b, &o.b.cache.?, kept);
            s.cached = kept.at;
        }
        s.fill_at.store(s.cached, .monotonic);
        return s.cached;
    }
    pub fn grow(_: Ops, _: *lanes.Stream, _: usize) !void {}
    pub fn body(o: Ops, s: *lanes.Stream, at: usize, rows: usize) Body {
        return .{ .b = o.b, .s = s, .at = at, .rows = rows };
    }
    pub fn finish(o: Ops, _: *lanes.Stream) !void {
        try o.b.drain();
    }
};
const Keep = struct {
    b: *Backend,
    saved: ?*snap.Snap = null,
    marks: usize = 0,
    failed: bool = false,
    fn mark(ptr: *anyopaque, _: *lanes.Stream, _: u32) void {
        const k: *Keep = @ptrCast(@alignCast(ptr));
        k.marks += 1;
    }
    fn interrupted(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *Keep = @ptrCast(@alignCast(ptr));
        if (at != boundary or k.b.completed != at or s.fill_at.load(.monotonic) != at or k.b.flights.items.len != 0) {
            k.failed = true;
            return;
        }
        k.saved = snap.save(k.b, &k.b.cache.?, at) catch {
            k.failed = true;
            return;
        };
    }
    fn yields(ptr: *anyopaque) bool {
        const b: *Backend = @ptrCast(@alignCast(ptr));
        return b.completed >= boundary;
    }
};
fn equalState(left: *Backend, right: *Backend) !void {
    const l = &left.cache.?;
    const r = &right.cache.?;
    try t.expectEqual(l.len, r.len);
    try t.expectEqual(l.mtp_len, r.mtp_len);
    const c = left.m.config;
    const conv_n = (c.conv_kernel - 1) * c.convDim() * 2;
    const ssm_n = c.mamba_heads * c.mamba_head_dim * c.state * 4;
    try t.expectEqualSlices(u8, left.pool.conv[0].data[l.slot * conv_n ..][0..conv_n], right.pool.conv[0].data[r.slot * conv_n ..][0..conv_n]);
    try t.expectEqualSlices(u8, left.pool.ssm[0].data[l.slot * ssm_n ..][0..ssm_n], right.pool.ssm[0].data[r.slot * ssm_n ..][0..ssm_n]);
    for (0..1 + @as(usize, @intFromBool(left.with_head))) |i| {
        const lk = if (i == 0) l.kv[0] else l.mtp.?;
        const rk = if (i == 0) r.kv[0] else r.mtp.?;
        const n = if (i == 0) l.len else l.mtp_len;
        for ([_]mtl.Buffer{ lk.k, lk.v }, [_]mtl.Buffer{ rk.k, rk.v }) |lb, rb| for (0..c.kv_heads) |h| {
            const off = h * lk.capacity * c.head_dim * 2;
            try t.expectEqualSlices(u8, lb.data[off..][0 .. n * c.head_dim * 2], rb.data[off..][0 .. n * c.head_dim * 2]);
        };
    }
}

test "Nemotron warm state: yielded unmarked prefix survives release and the follow-up restores its actual state" {
    var ids: [count]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i % 251 + 1);
    for ([_]bool{ false, true }) |head| {
        var model: Model = .{ .config = config() };
        var reference: Backend = .{ .m = &model, .pool = try st.Pool.init(a, model.device, model.config, 7, 1), .with_head = head };
        defer reference.deinit();
        var cold = try lanes.Stream.init(a, .{ .id = "cold", .prompt = &ids, .max_new = 1 });
        defer cold.deinit(a);
        try pass.whole(&reference, Ops{ .b = &reference }, &cold);
        try t.expectEqual(@as(u32, 0), cold.cached);
        try t.expectEqual(@as(usize, count), reference.processed);
        var b: Backend = .{ .m = &model, .pool = try st.Pool.init(a, model.device, model.config, 7, 1), .with_head = head };
        defer b.deinit();
        var keep: Keep = .{ .b = &b };
        defer if (keep.saved) |saved| snap.drop(&b, saved);
        {
            var warm = try lanes.Stream.init(a, .{ .id = "warm", .prompt = &ids, .max_new = 0, .chunks = &.{16384}, .reuse = .{
                .marks = &.{16384},
                .hook = .{ .ptr = &keep, .at = Keep.mark, .interrupted = Keep.interrupted },
            } });
            defer warm.deinit(a);
            warm.yield_check = .{ .ptr = &b, .check = Keep.yields };
            try t.expectError(error.Yielded, pass.whole(&b, Ops{ .b = &b }, &warm));
            try t.expectEqual(@as(usize, boundary), b.completed);
            try t.expectEqualSlices(u32, &.{boundary}, warm.prefill_widths.items);
            try t.expectEqual(@as(usize, 0), keep.marks);
            try t.expect(!keep.failed);
            try t.expect(keep.saved != null);
            try t.expectEqual(@as(u32, boundary), keep.saved.?.at);
            try b.release(); // drains the snapshot copy, then destroys all live KV buffers and gives back the slot
        }
        // Overwrite the released Mamba slots: a pointer to the old live state cannot satisfy the resumed comparison.
        @memset(b.pool.conv[0].data, 0xa5);
        @memset(b.pool.ssm[0].data, 0x5a);
        var follow = try lanes.Stream.init(a, .{ .id = "follow", .prompt = &ids, .max_new = 1, .reuse = .{ .saved = keep.saved.?, .at = boundary } });
        defer follow.deinit(a);
        try pass.whole(&b, Ops{ .b = &b }, &follow);
        try t.expectEqual(@as(u32, boundary), follow.cached);
        try t.expectEqual(@as(?usize, boundary), b.first);
        try t.expectEqual(@as(usize, count - boundary), b.processed);
        try equalState(&reference, &b);
    }
}
