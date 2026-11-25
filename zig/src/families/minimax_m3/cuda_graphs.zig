//! TF_MINIMAX_GRAPHS=1: a one-stream decode verify window as a CUDA graph, recorded each round and updated in place.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const gc = cuda.graph_cache;

/// The switch; default off. The startup check refused every malformed value already, so a plain default stands here.
pub fn on() bool {
    return @import("sys").environment.read(bool, "TF_MINIMAX_GRAPHS", false) catch false;
}

/// A window's launches, beside the arguments each refresh takes again: rows, sparse attention (index blocks), taps.
pub const Key = struct { rows: u32, sparse: bool, taps: bool };

/// A window a graph may hold: one decode segment, no prompt chunk, images, layer hook or tail cut.
pub fn eligible(f: *const fw.Forward, segs: []const fw.Seg) bool {
    return segs.len == 1 and !f.b.prefill and f.images == null and f.watch == null and f.tail == null;
}

pub fn keyOf(f: *const fw.Forward, rows: usize, seg: fw.Seg) Key {
    return .{ .rows = @intCast(rows), .sparse = f.sparse(seg), .taps = f.taps != null };
}

pub const Graphs = struct {
    state: ?*State = null, // set up at the first eligible window with the switch on (the exchange is known by then)
    off: bool = false,

    const State = struct { engine: gc.CudaEngine, cache: gc.Cache(Key), agreement: ?Agreement };

    pub fn deinit(g: *Graphs, gpa: std.mem.Allocator) void {
        if (g.state) |s| {
            s.cache.deinit();
            if (s.agreement) |*a| a.dev.free();
            gpa.destroy(s);
        }
        g.* = .{};
    }

    /// `f.run(ids, segs)`: with the switch on and an eligible window, its launches through the graph cache.
    pub fn run(g: *Graphs, gpa: std.mem.Allocator, f: *const fw.Forward, ids: []const i32, segs: []const fw.Seg) !void {
        if (g.off or !eligible(f, segs)) return f.run(ids, segs);
        if (g.state == null and !on()) { // read once, at the first eligible window
            g.off = true;
            return f.run(ids, segs);
        }
        const s = g.state orelse try g.setUp(gpa, f);
        try f.upload(ids, segs);
        var body: Body = .{ .f = f, .rows = ids.len, .segs = segs };
        const fp = gc.fingerprintOf(&.{ f.b.ids, f.b.resid, f.b.logits, @as(u64, f.b.rows) });
        _ = try s.cache.run(keyOf(f, ids.len, segs[0]), f.stream, fp, .{ .ctx = &body, .run = Body.run });
    }

    fn setUp(g: *Graphs, gpa: std.mem.Allocator, f: *const fw.Forward) !*State {
        const s = try gpa.create(State);
        errdefer gpa.destroy(s);
        s.engine = .{ .d = f.d };
        s.agreement = switch (f.exchange) {
            .local => null,
            .nccl => |n| .{ .lib = n.lib, .comm = n.comm, .stream = f.stream, .dev = try cuda.DeviceBuffer.alloc(f.d, 16) },
        };
        s.cache = gc.Cache(Key).init(gpa, s.engine.refreshing(), .{ .on = true, .max = 48 });
        s.cache.name = "minimax verify";
        if (s.agreement) |*a| s.cache.agree = .{ .ctx = a, .all = Agreement.all };
        g.state = s;
        return s;
    }
};

/// The launches a graph records: the window's whole forward past its ids' upload.
const Body = struct {
    f: *const fw.Forward,
    rows: usize,
    segs: []const fw.Seg,

    fn run(ctx: *anyopaque, _: cuda.Stream) anyerror!void {
        const b: *Body = @ptrCast(@alignCast(ctx));
        try b.f.launches(b.rows, b.segs);
    }
};

/// Every rank's verdict: one int's minimum over the ranks, read back (outside any capture).
const Agreement = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    dev: cuda.DeviceBuffer,

    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const a: *Agreement = @ptrCast(@alignCast(ctx));
        var v: i32 = @intFromBool(ok);
        try a.dev.uploadAsync(0, std.mem.asBytes(&v), a.stream.handle);
        try a.lib.check(a.lib.api.ncclAllReduce(a.dev.ptr, a.dev.ptr, 1, .i32, .min, a.comm, a.stream.handle), "ncclAllReduce");
        try a.stream.synchronize();
        try a.dev.download(0, std.mem.asBytes(&v));
        return v == 1;
    }
};

test "MiniMax graphs: the key separates rows, sparse attention and taps; the same launches share one" {
    const t = std.testing;
    const keys = [_]Key{
        .{ .rows = 1, .sparse = false, .taps = false }, .{ .rows = 4, .sparse = false, .taps = false },
        .{ .rows = 4, .sparse = true, .taps = false },  .{ .rows = 4, .sparse = true, .taps = true },
        .{ .rows = 8, .sparse = true, .taps = true },
    };
    for (keys, 0..) |a, i| for (keys, 0..) |b, j| try t.expectEqual(i == j, std.meta.eql(a, b));
}
