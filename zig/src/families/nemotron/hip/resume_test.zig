//! Resumes through Nemotron's real prompt cache, page swaps and prompt chunks; the MTP head's rows kept on the host.
const std = @import("std");
const hip = @import("hip");
const serve = @import("hip_serve");
const lanes = @import("lanes.zig");
const prompt_mod = @import("prompt.zig");
const Engine = @import("engine.zig").Engine;
const forward = @import("forward.zig");
const model = @import("model.zig");
const pages = @import("pages.zig");
const prefix = @import("prefix.zig");
const T = std.testing;

/// Pages in the pool, and the positions a stream holds.
const count = 64;
const cap = 32 * pages.tokens;

const Host = struct {
    fn upload(dst: hip.abi.DevicePtr, src: [*]const u8, n: usize) callconv(.c) hip.abi.Result {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], src[0..n]);
        return 0;
    }

    fn alloc(out: *hip.abi.DevicePtr, n: usize) callconv(.c) hip.abi.Result {
        out.* = std.c.malloc(@max(n, 1)) orelse return 1;
        return 0;
    }

    fn free(ptr: hip.abi.DevicePtr) callconv(.c) hip.abi.Result {
        std.c.free(ptr);
        return 0;
    }
};

/// The head's K/V rows by page and row: the token each position was absorbed with, as head.zig writes them.
const Rows = struct {
    at: [count * pages.tokens]u32 = @splat(0),

    pub fn absorb(r: *Rows, _: hip.ops.Ops, s: *pages.Stream, _: hip.ops.Tensor, next: []const u32) !void {
        for (next, s.mtp_len..) |t, pos| r.at[s.ids.items[pos / pages.tokens] * pages.tokens + pos % pages.tokens] = t;
        s.mtp_len += next.len;
    }

    fn of(r: *const Rows, s: *const pages.Stream, pos: usize) u32 {
        return r.at[s.ids.items[pos / pages.tokens] * pages.tokens + pos % pages.tokens];
    }
};

/// The engine the prompt chunk sees: no kernels run.
const Inert = struct {
    stream: hip.Stream,
    o: struct { world: usize = 1 } = .{},
    act: model.Kind = .bf16,
    m: model.Model = undefined,
    pool: pages.Pool = undefined,
    arena: hip.Arena = undefined,

    pub fn fresh(_: *Inert) !void {}

    pub fn ops(_: *Inert, _: anytype) hip.ops.Ops {
        return .{ .l = undefined, .bf16 = true, .stream = hip.counting, .arena = undefined };
    }

    pub fn ids(_: *Inert, _: hip.ops.Ops, _: []const u32) !u64 {
        return 0;
    }

    pub fn model_(e: *Inert) *const model.Model {
        return &e.m;
    }

    pub const poll = Engine.poll;

    pub fn logits(_: *Inert, _: hip.ops.Ops, x: hip.ops.Tensor, _: usize) !hip.ops.Tensor {
        return x;
    }
};

const Fixture = struct {
    runtime: hip.Runtime,
    /// The cache's side reads only the engine's driver, stream and pool; the chunk runs on `e`.
    engine: Engine,
    inert: Inert,
    e: *Inert,
    rows: Rows = .{},
    head: ?*Rows,
    side: prefix.Side,
    tp: serve.protocol.Sender,
    cache: *prefix.Prefix,

    fn init(x: *Fixture, head: bool) !void {
        const a = T.allocator;
        x.runtime = .{ .lib = undefined, .api = undefined };
        x.runtime.api.hipMemcpyHtoD = Host.upload;
        x.runtime.api.hipMalloc = Host.alloc;
        x.runtime.api.hipFree = Host.free;
        x.engine = undefined;
        x.engine.driver = x.runtime;
        x.engine.stream = .{ .r = &x.runtime, .handle = hip.counting };
        x.engine.pool = .{
            .gpa = a,
            .count = count,
            .kv_heads = 0,
            .head_kv = null,
            .head_dim = 0,
            .act = .bf16,
            .keys = &.{},
            .values = &.{},
            .ids = try serve.pages.Ids.init(a, count),
        };
        errdefer x.engine.pool.ids.deinit(a);
        x.inert = .{ .stream = x.engine.stream };
        x.inert.m.dims.hidden = 1;
        x.e = &x.inert;
        x.rows = .{};
        x.head = if (head) &x.rows else null;
        x.side = .{ .e = &x.engine, .state_bytes = 8 };
        x.tp = .{ .gpa = a };
        x.cache = try prefix.init(a, &x.side, &x.tp, head);
        x.cache.keepPages(count, 8);
    }

    fn deinit(x: *Fixture) void {
        x.cache.deinit();
        x.engine.pool.ids.deinit(T.allocator);
    }

    fn lane(x: *Fixture) !lanes.Lane {
        const desc = try hip.DeviceBuffer.alloc(&x.runtime, pages.Stream.descBytes(0, cap));
        const mem = try hip.DeviceBuffer.alloc(&x.runtime, 8);
        const s: pages.Stream = .{ .gpa = T.allocator, .desc = desc, .table_at = 0, .cap = cap, .mem = mem, .states = &.{} };
        return .{ .s = s, .snaps = undefined, .hidden = try hip.DeviceBuffer.alloc(&x.runtime, 2) };
    }

    fn drop(x: *Fixture, l: *lanes.Lane) void {
        l.hidden.free();
        l.s.deinit(&x.engine.pool);
    }

    /// One prompt pass as the lanes run it: the cache's resume and marks, a chunk to each mark and a keep there.
    fn pass(x: *Fixture, l: *lanes.Lane, prompt: []const u32) !usize {
        const a = T.allocator;
        var adopt: std.ArrayList(u32) = .empty;
        defer adopt.deinit(a);
        const plan = try x.cache.begin(prompt, null, 0, &.{}, l, &adopt);
        defer a.free(plan.marks);
        if (adopt.items.len > 0) try l.s.adoptRollback(&x.engine.pool, &adopt);
        if (plan.from > 0) l.s.resumeAt(plan.from);
        var at = plan.from;
        for (plan.marks) |m| {
            try l.s.cover(&x.engine.pool, m);
            _ = try prompt_mod.chunk(x, l, prompt, at, m, null);
            try x.cache.keep(l, prompt, null, m);
            at = m;
        }
        try l.s.cover(&x.engine.pool, prompt.len);
        _ = try prompt_mod.chunk(x, l, prompt, at, prompt.len, null);
        return plan.from;
    }

    pub fn runPrompt(_: *Fixture, _: hip.ops.Ops, ctx: forward.Ctx, _: u64, _: usize, pos: usize) !hip.ops.Tensor {
        try T.expectEqual(pos, ctx.stream.len);
        return .{ .ptr = 1, .kind = .bf16 };
    }
};

/// Two 520-token prompts, the same through token 511 and apart at 512, where the cache keeps a state.
fn prompts() [2][520]u32 {
    var p: [2][520]u32 = undefined;
    for (&p[0], &p[1], 0..) |*x, *y, i| {
        x.* = @intCast(100 + i);
        y.* = if (i < 512) x.* else @intCast(9000 + i);
    }
    return p;
}

test "Nemotron HIP resume: a prompt parting at a kept state never rewrites the head's rows another stream holds" {
    var x: Fixture = undefined;
    try x.init(true);
    defer x.deinit();
    const p = prompts();
    var first = try x.lane();
    defer x.drop(&first);
    try T.expectEqual(@as(usize, 0), try x.pass(&first, &p[0]));
    var second = try x.lane();
    defer x.drop(&second);
    _ = try x.pass(&second, &p[1]);
    // the first stream's pairs are its own, row 511 with its token 512
    for (0..p[0].len - 1) |pos| try T.expectEqual(p[0][pos + 1], x.rows.of(&first.s, pos));
    for (0..p[1].len - 1) |pos| try T.expectEqual(p[1][pos + 1], x.rows.of(&second.s, pos));
    // the same prompt again resumes at the kept state, on the first stream's pages, with nothing left to absorb there
    var again = try x.lane();
    defer x.drop(&again);
    try T.expectEqual(@as(usize, 512), try x.pass(&again, &p[0]));
    try T.expectEqual(first.s.ids.items[7], again.s.ids.items[7]);
    for (0..p[0].len - 1) |pos| try T.expectEqual(p[0][pos + 1], x.rows.of(&first.s, pos));
    for (0..p[0].len - 1) |pos| try T.expectEqual(p[0][pos + 1], x.rows.of(&again.s, pos));
}

test "Nemotron HIP resume: without the head a prompt parting at a kept state still resumes there" {
    var x: Fixture = undefined;
    try x.init(false);
    defer x.deinit();
    const p = prompts();
    var first = try x.lane();
    defer x.drop(&first);
    _ = try x.pass(&first, &p[0]);
    var second = try x.lane();
    defer x.drop(&second);
    try T.expectEqual(@as(usize, 512), try x.pass(&second, &p[1]));
    try T.expectEqual(first.s.ids.items[7], second.s.ids.items[7]);
}
