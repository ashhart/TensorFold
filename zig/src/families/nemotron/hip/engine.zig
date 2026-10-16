//! Nemotron-H on one HIP device: the kernels, a stream, the rank's weights, the KV pool and the scratch.
const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const model = @import("model.zig");
const load = @import("load.zig");
const pages = @import("pages.zig");
const plan = @import("plan.zig");
const admit = @import("admit.zig");
const forward = @import("forward.zig");

pub const Options = struct {
    device: c_int = 0,
    rank: usize = 0,
    world: usize = 1,
    group: ?hip.collective.Group = null,
    policy: hip.Policy = .{},
    /// Positions a stream holds at most.
    capacity: usize = 4096,
    /// Prompt rows a forward takes at most.
    chunk: usize = 512,
    /// Streams the KV pool holds at the full capacity.
    streams: usize = 1,
    /// Rank 0 loads the MTP head when its file is there.
    mtp: bool = true,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: hip.Runtime,
    ctx: hip.Context,
    lib: hip.Launcher,
    stream: hip.Stream,
    o: Options,
    cfg: config.Config,
    weights: load.Weights,
    act: model.Kind,
    comm: ?hip.collective.Comm = null,
    arena: hip.Arena,
    pool: pages.Pool,
    /// Each block's place among its kind; the MTP head's attention takes the pool slot after the model's.
    place: [config.max_layers]usize,
    mtp_slot: usize,

    /// The checkpoint `dir` (admitted before) on device `o.device`, this rank's share loaded.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.io = io;
        e.o = o;
        e.driver = try hip.Runtime.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, o.device);
        errdefer e.ctx.deinit();
        const caps = try hip.Device.kernelCaps(&e.driver, o.device);
        e.lib = try hip.Launcher.load(&e.driver, o.policy.choices(), hip.kernels.images);
        errdefer e.lib.unload();
        e.act = if (caps.act == .f16) .f16 else .bf16;
        e.stream = try hip.Stream.init(&e.driver);
        errdefer e.stream.deinit();
        e.comm = null;
        if (o.world > 1) e.comm = try (o.group orelse return error.NoGroup).open();
        errdefer if (e.comm) |c| c.close();
        e.cfg = try config.Config.read(gpa, io, dir);
        const mtp = o.mtp and o.rank == 0 and hasFile(gpa, dir, tensors.mtp_file);
        e.weights = try load.load(
            gpa,
            io,
            &e.driver,
            dir,
            &e.cfg,
            .{ .world = o.world, .rank = o.rank, .mtp = mtp, .act = e.act },
        );
        errdefer e.weights.deinit();
        e.weights.model.tp = e.comm;
        e.arena = try hip.Arena.init(&e.driver, admit.scratch_bytes);
        errdefer e.arena.deinit();
        const d = e.weights.model.dims;
        e.pool = try pages.Pool.init(
            gpa,
            &e.driver,
            plan.attentionLayers(&e.cfg, mtp),
            o.streams * pages.pagesFor(o.capacity),
            d.kv_heads,
            d.head_dim,
            e.act,
        );
        _ = forward.slots(&e.cfg, &e.place);
        e.mtp_slot = e.cfg.count(.attention);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.pool.deinit();
        e.arena.deinit();
        e.weights.deinit();
        if (e.comm) |c| c.close();
        e.stream.deinit();
        e.lib.unload();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    pub fn model_(e: *const Engine) *const model.Model {
        return &e.weights.model;
    }

    pub fn ops(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
        return .{ .l = &e.lib, .bf16 = e.act == .bf16, .stream = e.stream.handle, .arena = arena };
    }

    /// A new stream at the engine's capacity: its page table and zeroed Mamba states.
    pub fn openStream(e: *Engine) !pages.Stream {
        return pages.Stream.init(e.gpa, &e.driver, e.weights.model.dims, e.act, e.cfg.count(.mamba), e.o.capacity);
    }

    pub fn closeStream(e: *Engine, s: *pages.Stream) void {
        e.stream.synchronize() catch {};
        s.deinit(&e.pool);
    }

    /// `tokens` as int32 ids in fresh arena memory, the copy queued on the stream.
    pub fn ids(e: *Engine, o: hip.ops.Ops, tokens: []const u32) !u64 {
        const dst = try o.arena.of(i32, tokens.len);
        try hip.runtime.check(e.driver.api.hipMemcpyHtoDAsync(
            @ptrFromInt(dst),
            @ptrCast(@constCast(tokens.ptr)),
            tokens.len * 4,
            o.stream,
        ));
        try e.stream.synchronize();
        return dst;
    }

    /// A prompt's `tokens` from `s.len` into `s`, kept: the final-normed rows of its last chunk (valid until the next).
    pub fn prompt(e: *Engine, s: *pages.Stream, tokens: []const u32) !hip.ops.Tensor {
        var out: hip.ops.Tensor = undefined;
        var at: usize = 0;
        while (at < tokens.len) {
            const n = @min(e.o.chunk, tokens.len - at);
            try s.cover(&e.pool, s.len + n);
            try e.stream.synchronize();
            e.arena.reset();
            const o = e.ops(&e.arena);
            const dev = try e.ids(o, tokens[at..][0..n]);
            out = try forward.run(
                o,
                &e.weights.model,
                e.place[0..e.cfg.layers],
                .{ .pool = &e.pool, .stream = s },
                dev,
                n,
                s.len,
                true,
            );
            s.len += n;
            at += n;
        }
        return out;
    }

    /// The output head over `rows` final rows: whole logits rows (act dtype), joined across ranks in rank order.
    pub fn logits(e: *Engine, o: hip.ops.Ops, normed: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        const m = &e.weights.model;
        const y = try o.project(normed, m.head, rows, false);
        const c = m.tp orelse return y;
        const width = m.head.n;
        const size = y.kind.size();
        const parts = try o.arena.take(c.world * rows * width * size);
        try hip.reduce.gather(o, c, y, parts, rows * width);
        const out = try o.arena.take(c.world * rows * width * size);
        for (0..rows) |row| for (0..c.world) |r| {
            const to = out + (row * c.world + r) * width * size;
            const from = parts + (r * rows + row) * width * size;
            try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(
                @ptrFromInt(to),
                @ptrFromInt(from),
                width * size,
                o.stream,
            ));
        };
        return .{ .ptr = out, .kind = y.kind };
    }

    /// The MTP head over `next` (the tokens after the rows of `hidden`) at its next positions: draft logits.
    pub fn draft(
        e: *Engine,
        o: hip.ops.Ops,
        s: *pages.Stream,
        next: []const u32,
        hidden: hip.ops.Tensor,
        prompt_rows: bool,
    ) !hip.ops.Tensor {
        const dev = try e.ids(o, next);
        const ctx: forward.Ctx = .{ .pool = &e.pool, .stream = s };
        const y = try forward.draft(
            o,
            &e.weights.model,
            e.mtp_slot,
            ctx,
            dev,
            hidden,
            next.len,
            s.mtp_len,
            prompt_rows,
        );
        s.mtp_len += next.len;
        return y;
    }
};

/// Whether `dir`/`name` exists.
pub fn hasFile(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) bool {
    const path = std.fs.path.joinZ(gpa, &.{ dir, name }) catch return false;
    defer gpa.free(path);
    return std.c.access(path, std.c.F_OK) == 0;
}
