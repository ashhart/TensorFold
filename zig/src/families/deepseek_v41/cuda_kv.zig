//! The forward's KV in the paged pool: comp and index-key families in kv.DevicePool, split KV and its exchanges.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const tp = @import("tp");
const kv = @import("kv");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const Config = @import("config.zig").Config;

const sessions = kv.sessions;

pub const Error = error{ PoolTooSmall, BadGlue, NoPool };

pub const Options = struct {
    /// TF_DSV41_POOL_TOKENS (whole pages of every residue)
    tokens: u64,
    /// TF_DSV41_KV_SPLIT (needs TP > 1)
    split: bool = false,
    /// the live slot's capacity in positions (Forward's limit)
    capacity: u64,
    /// TF_DSV41_KV_SPLIT_UNION_MIB, bytes
    union_cap: u64 = 128 << 20,
    /// live slots (slots.zig, TF_DSV41_SLOTS): the pool's slots and the device tables' rows
    slots: u32 = 1,
    /// bytes the plan holds resident beyond the measured terms (the long-prompt stream scratch); added to boot growth
    resident: u64 = 0,
    /// TF_DSV41_PF_4K / TF_DSV41_PF_TBO: a long prompt's prefill workspace; worst MemAvailable minus it clears floor
    workspace: u64 = 0,
};

/// TF_DSV41_POOL_TOKENS / TF_DSV41_KV_SPLIT / TF_DSV41_KV_SPLIT_UNION_MIB (null: no pool, the contiguous slot).
pub fn optionsFromEnv(capacity: u64, world: u32) !?Options {
    const t = std.c.getenv("TF_DSV41_POOL_TOKENS") orelse return null;
    const tokens = try std.fmt.parseInt(u64, std.mem.span(t), 10);
    const split = if (std.c.getenv("TF_DSV41_KV_SPLIT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    if (split and world < 2) return error.SplitNeedsTp;
    const mib: u64 = if (std.c.getenv("TF_DSV41_KV_SPLIT_UNION_MIB")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else 128;
    // pool settings: rounded up to whole pages of every residue
    const unit: u64 = 256 * @as(u64, world);
    return .{ .tokens = (tokens + unit - 1) / unit * unit, .split = split, .capacity = capacity, .union_cap = mib << 20 };
}

/// The boot check's figures for the measured check at the end of boot (`afterBoot`).
pub const Boot = struct { rank: u32, at_check: f64, pool_gib: f64, pool_tokens: u64, bytes_a_token: f64, reserve: f64, dip: f64, hard: f64, target: f64, workspace: u64 };

/// Before the pool (GB10): refuses a pool that cannot fit at all; growth is measured at boot's end (null: not GB10).
pub fn bootCheck(ctx: *const cuda.Context, cfg: *const Config, o: Options, rank: u32, world: u32) !?Boot {
    const budget = kv.budget;
    const mi = budget.meminfo() catch |e| {
        if (!quiet) std.log.warn("kv: no /proc/meminfo ({t}): the measured boot check is skipped", .{e});
        return null;
    };
    const dev = try ctx.memInfo();
    const integrated = (ctx.attribute(.integrated) catch 0) != 0;
    if (!budget.measuredApplies(budget.modeFromEnv(), integrated, dev.total, mi.total)) return null;
    var knobs: budget.Knobs = .{};
    if (std.c.getenv("TF_DSV41_FLOOR_HARD_GIB")) |v| knobs.hard_gib = try std.fmt.parseFloat(f64, std.mem.span(v));
    // TF_DSV41_FLOOR_GIB (5): the floor target the load plans for (0 < hard <= target)
    const target: f64 = if (std.c.getenv("TF_DSV41_FLOOR_GIB")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else @max(5.0, knobs.hard_gib);
    if (!(knobs.hard_gib > 0 and knobs.hard_gib <= target)) {
        std.log.err("kv: TF_DSV41_FLOOR_HARD_GIB={d} / TF_DSV41_FLOOR_GIB={d}: expected 0 < hard <= target", .{ knobs.hard_gib, target });
        return error.BadFloor;
    }
    var layout = try kv.Layout.fromConfig(cfg, .{ .split = o.split and world > 1 });
    // Zig's terms, not the reference's: its plan holds from boot what a caching allocator takes while serving
    var terms = try budget.zigFromEnv();
    terms.growth += @as(f64, @floatFromInt(o.resident)) / budget.GiB;
    const r = budget.serveCheck(knobs, .{ .rank = rank, .avail_gib = @as(f64, @floatFromInt(mi.available)) / budget.GiB, .terms = terms }, layout.pool(), world, o.tokens);
    var buf: [1024]u8 = undefined;
    if (!r.ok) {
        std.log.err("kv: {s}", .{r.message(&buf)});
        return error.PoolDoesNotFit;
    }
    std.log.info("kv: before the pool: {s}; the boot's growth is measured at its end", .{r.message(&buf)});
    return .{ .rank = rank, .at_check = r.avail_gib, .pool_gib = r.pool_gib, .pool_tokens = o.tokens, .bytes_a_token = r.bytes_a_token, .reserve = try budget.reserveFromEnv(), .dip = terms.dip, .hard = r.hard, .target = target, .workspace = o.workspace };
}

/// At boot's end: idle - dip >= the hard floor or the pool is refused; a 4K workspace past it warns (wsAgreed).
pub fn afterBoot(b: Boot) !void {
    const budget = kv.budget;
    const idle = @as(f64, @floatFromInt((try budget.meminfo()).available)) / budget.GiB;
    const x = budget.afterBoot(b.at_check, b.pool_gib, idle, b.reserve, b.dip, b.hard, b.pool_tokens, b.bytes_a_token);
    const msg = "kv: rank {d}: boot growth {d:.2} GiB measured (MemAvailable {d:.2} at the check - pool {d:.2} - {d:.2} now); worst serving point {d:.2} GiB (post-boot reserve {d:.2}, TF_DSV41_POST_BOOT_GIB; dip {d:.2}, TF_DSV41_SERVE_DIP_GIB; hard floor {d:.1})";
    const args = .{ b.rank, x.growth, b.at_check, b.pool_gib, idle, x.worst, b.reserve, b.dip, b.hard };
    if (!x.ok) {
        std.log.err(msg ++ ": the pool does not fit; the largest that does: TF_DSV41_POOL_TOKENS={d}", args ++ .{x.largest});
        return error.PoolDoesNotFit;
    }
    std.log.info(msg, args);
    if (b.workspace > 0) {
        const ws = @as(f64, @floatFromInt(b.workspace)) / budget.GiB;
        const left = x.worst - ws;
        if (left < b.hard) if (!quiet) std.log.warn("kv: rank {d}: the prefill workspace ({d:.2} GiB, TF_DSV41_PF_4K / TF_DSV41_PF_TBO) at the worst point leaves {d:.2} GiB, under the {d:.1} GiB hard floor: long prompts there run 2,048-row segments", .{ b.rank, ws, left, b.hard });
    }
    if (x.worst < b.target) if (!quiet) std.log.warn("kv: memory: floor {d:.2} GiB is under the {d} GiB target (TF_DSV41_FLOOR_GIB)", .{ x.worst, b.target });
}

pub const Kv = struct {
    gpa: std.mem.Allocator,
    layout: kv.Layout,
    pool: sessions.Pool,
    dev: kv.DevicePool,
    /// the current slot (one-slot operations), one of `slots`
    slot: *sessions.Slot,
    slots: []*sessions.Slot = &.{},
    opts: Options,
    world: u32,
    rank: u32,
    comm: tp.collective.Collective,
    /// the forward's runner (kernels, stream) for the exchange's copies
    runner: *run.Runner,
    /// a union's device list of local rows to send (grows)
    phys: ?cuda.DeviceBuffer = null,
    /// the exchange (kv/split.zig, dense or TF_DSV41_KV_SPLIT_COMPACT) and its persistent int32 [W] owner lengths
    x: kv.split.Exchange = undefined,
    ops: Ops = undefined,
    lens: ?cuda.DeviceBuffer = null,
    /// TF_DSV41_KV_SPLIT_DEDUP's map and scratch (kv/split.zig Dedup)
    dedup: ?cuda.DeviceBuffer = null,
    /// TF_DSV41_KV_SPLIT_HISTORY: a union that fits the cap sends the slot's owned rows (kv/split.zig HistArgs)
    hist: bool = false,
    stats: Stats = .{},
    /// the last union's row blocks when it took several (empty: one block); longpf.zig exchanges them block by block
    blocks: std.ArrayList(kv.split.Block) = .empty,
    block_phys: std.ArrayList(u32) = .empty,
    union_at: struct { comp: u64 = 0, send: u64 = 0, recv: u64 = 0 } = .{},
    /// a union's pinned staging: the selection read back, the tokens and rows sent up (reused after the next drain)
    pin_sel: ?cuda.HostBuffer = null,
    pin_up: ?cuda.HostBuffer = null,

    pub const Stats = struct { dense: u64 = 0, unions: u64 = 0, union_rows_max: u64 = 0, bytes: u64 = 0 };

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, cfg: *const Config, o: Options, runner: *run.Runner, comm: tp.collective.Collective) !*Kv {
        const world: u32 = @intCast(comm.world());
        const k = try gpa.create(Kv);
        errdefer gpa.destroy(k);
        k.* = .{ .gpa = gpa, .layout = try kv.Layout.fromConfig(cfg, .{ .split = o.split and world > 1 }), .pool = undefined, .dev = undefined, .slot = undefined, .opts = o, .world = world, .rank = @intCast(comm.rank()), .comm = comm, .runner = runner };
        k.pool = try sessions.Pool.init(gpa, k.layout.pool(), o.tokens, if (k.layout.opts.split) world else 1, if (k.layout.opts.split) k.rank else 0);
        errdefer k.pool.deinit();
        k.slots = try gpa.alloc(*sessions.Slot, @max(o.slots, 1));
        errdefer gpa.free(k.slots);
        for (k.slots) |*s| s.* = try k.pool.newSlot(o.capacity);
        k.slot = k.slots[0];
        k.dev = try kv.DevicePool.init(gpa, d, ctx, &k.pool, runner.stream, @intCast(k.slots.len), k.slot.max_pages);
        errdefer k.dev.deinit();
        k.ops = .{ .r = runner };
        if (k.layout.opts.split) {
            k.lens = try cuda.DeviceBuffer.alloc(d, 4 * @as(usize, world));
            // the compact mode's transport agreement happens here, once, outside any graph
            k.x = try kv.split.Exchange.init(comm, k.ops.kernels(), try kv.split.compactMode());
            if (kv.split.dedupOn()) try k.initDedup(d, cfg.index_topk);
            if (kv.split.historyOn()) {
                if (runner.kernels.ops.kv_hist == null) {
                    std.log.err("[tensorfold] TF_DSV41_KV_SPLIT_HISTORY=1: the kvsplit fatbin has no history kernels (rebuild it)", .{});
                    return error.MissingKernel;
                }
                k.hist = true;
            }
        }
        return k;
    }

    /// The dedup scratch: a key map over every slot's rows (ratio 1), INT_MAX; entries for 64 rows (wider: no dedup).
    fn initDedup(k: *Kv, d: *const cuda.Driver, topk: u32) !void {
        if (k.runner.kernels.ops.kv_dedup == null) {
            std.log.err("[tensorfold] TF_DSV41_KV_SPLIT_DEDUP=1: the kvsplit fatbin has no dedup kernels (rebuild it)", .{});
            return error.MissingKernel;
        }
        const slot_rows = @as(u64, k.slot.max_pages) * block.Pool.page;
        const map_rows = slot_rows * k.slots.len;
        const entries = 64 * @as(u64, topk);
        const bcnt = (entries + 255) / 256 * kv.split.max_world;
        const buf = try cuda.DeviceBuffer.alloc(d, @intCast(4 * (map_rows + 2 * entries + bcnt)));
        k.dedup = buf;
        try buf.fill32(std.math.maxInt(i32), null);
        k.x.dedup = .{ .map = buf.ptr, .map_rows = map_rows, .first = buf.ptr + 4 * map_rows, .place = buf.ptr + 4 * (map_rows + entries), .bcnt = buf.ptr + 4 * (map_rows + 2 * entries), .entries = entries, .slots = @intCast(k.slots.len) };
        if (k.rank == 0) std.log.info("[tensorfold] split KV dedup: {d} MiB (map {d} rows, {d} entries)", .{ buf.len >> 20, map_rows, entries });
    }

    pub fn close(k: *Kv) !void {
        try k.runner.stream.synchronize();
        try k.dev.io.synchronize();
        k.deinit();
    }
    fn deinit(k: *Kv) void {
        k.dropBlocks();
        k.blocks.deinit(k.gpa);
        k.block_phys.deinit(k.gpa);
        if (k.phys) |*b| b.free();
        if (k.lens) |*b| b.free();
        if (k.dedup) |*b| b.free();
        if (k.pin_sel) |*b| b.free();
        if (k.pin_up) |*b| b.free();
        k.dev.deinit();
        k.pool.deinit();
        k.gpa.free(k.slots);
        k.gpa.destroy(k);
    }

    /// Slot i becomes the current one (slots.zig: one-slot operations on slot i).
    pub fn select(k: *Kv, i: u32) void {
        k.slot = k.slots[i];
    }

    /// The emitter's view of the pool (block.Options.pool).
    pub fn options(k: *Kv) block.Pool {
        const fams = k.layout.families();
        return .{
            .comp_pages = k.pool.familyTensorPages(fams[0]),
            .ik_pages = k.pool.familyTensorPages(fams[1]),
            .pts = k.slot.max_pages,
            .split = k.layout.opts.split,
            .early = k.layout.opts.split and kv.split.earlyOn(),
            .union_cap = @intCast(k.opts.union_cap),
        };
    }

    /// Points the window's pool roles at the device pool's buffers (before Runner.bind), checking their coverage.
    pub fn bind(k: *Kv, r: *run.Runner, plan: *const buffers.Plan) !void {
        var nb: [64]u8 = undefined;
        for (k.layout.sources[0..k.layout.nsources]) |L| {
            const c = k.layout.compOf(L).?;
            try k.external(r, plan, try std.fmt.bufPrint(&nb, "s.kv.comp.L{d}", .{L}), k.dev.tensors[c]);
            try k.external(r, plan, try std.fmt.bufPrint(&nb, "s.kv.ik.L{d}", .{L}), k.dev.tensors[c + 1]);
        }
        try k.external(r, plan, "s.kv.pt", k.dev.table);
        try k.external(r, plan, "s.kv.ct", k.dev.local);
    }

    fn external(_: *Kv, r: *run.Runner, plan: *const buffers.Plan, role: []const u8, b: cuda.DeviceBuffer) !void {
        if (plan.sizes.get(role)) |need| if (need > b.len) return error.PoolTooSmall;
        try r.external(role, b.ptr);
    }

    /// Maps every page holding positions < end and uploads the changed table entries (before a window's launches).
    pub fn reserve(k: *Kv, end: u64) !void {
        try k.slot.ensure(end);
        try k.dev.syncTables();
    }

    /// Keeps the pages holding positions < keep (a rollback, or a release at 0).
    pub fn truncate(k: *Kv, keep: u64) !void {
        _ = try k.slot.truncate(keep);
        try k.dev.syncTables();
    }

    /// The slot's pages back to the pool (a session entry may still hold them).
    pub fn release(k: *Kv) !void {
        _ = try k.slot.releaseAll();
        try k.dev.syncTables();
    }

    // -- split KV's exchange (block.zig Emitter.exchange: sel, comp family, split table, send, recv, tokens, psh) --

    pub fn glue(k: *Kv, r: *run.Runner, c: *const calls.Call, step: []const u8) !void {
        k.dropBlocks();
        if (std.mem.eql(u8, step, "kx_dense")) return k.dense(r, c);
        if (std.mem.eql(u8, step, "kx_union")) return k.unionOf(r, c);
        return error.BadGlue;
    }

    fn args(r: *run.Runner, c: *const calls.Call) !struct { sel: calls.Tensor, comp: u64, table: u64, send: u64, recv: u64, tok: u64, psh: u32 } {
        const x = c.args;
        if (x.len != 7) return error.BadGlue;
        return .{
            .sel = x[0].arg.t,
            .comp = try r.tensorAddr(x[1].arg.t),
            .table = try r.tensorAddr(x[2].arg.t),
            .send = try r.tensorAddr(x[3].arg.t),
            .recv = try r.tensorAddr(x[4].arg.t),
            .tok = try r.tensorAddr(x[5].arg.t),
            .psh = @intCast(x[6].arg.i),
        };
    }

    /// Every rank sends its owned selected rows (a discard row elsewhere) in one all-gather; token owner x R x K + j.
    fn dense(k: *Kv, r: *run.Runner, c: *const calls.Call) !void {
        const a = try args(r, c);
        const rows: u32 = @intCast(a.sel.shape[0]);
        const kk: u32 = @intCast(a.sel.shape[1]);
        const da: kv.split.DenseArgs = .{
            .sel = try r.tensorAddr(a.sel),
            .rows = rows,
            .k = kk,
            .table = a.table,
            .pts = 0,
            .rslot = 0,
            .psh = a.psh,
            .base = a.comp,
            .row_bytes = block.Pool.row_bytes,
            .world = k.world,
            .send = a.send,
            .tok = a.tok,
        };
        const bytes = @as(usize, rows) * kk * block.Pool.row_bytes;
        // packed where the compact mode and the transport take it (the same answer on every rank), else dense
        try k.x.window(kv.split.PackArgs.of(da, k.rank, k.lens.?.ptr), a.recv, r.stream.handle);
        k.stats.dense += 1;
        k.stats.bytes += bytes * k.world;
    }

    /// A prefill segment's union: sorted unique selected rows, padded per owner (one host sync), token owner x M + i.
    fn unionOf(k: *Kv, r: *run.Runner, c: *const calls.Call) !void {
        const a = try args(r, c);
        const n: usize = @intCast(a.sel.shape[0]);
        const kk: u32 = @intCast(a.sel.shape[1]);
        if (k.hist) {
            const m = kv.split.historyRows(k.slot.len, a.psh, k.world);
            if (m > 0 and m <= k.opts.union_cap / ((@as(u64, k.world) + 1) * block.Pool.row_bytes)) return k.historyOf(r, a, n, kk, m);
        }
        // the drain also ends the last union's async uploads, so its pinned staging is free again
        try r.stream.synchronize();
        const sel = (try pinned(r.d, &k.pin_sel, n * kk * 4)).slice(i32)[0 .. n * kk];
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(a.sel), .len = sel.len * 4 }, 0, std.mem.sliceAsBytes(sel));
        const table = try k.gpa.alloc(u32, k.slot.max_pages);
        defer k.gpa.free(table);
        for (table, 0..) |*t, i| t.* = k.slot.localTableAt(@intCast(i));
        var blocks: std.ArrayList(kv.split.Block) = .empty;
        defer {
            for (blocks.items) |*b| b.deinit(k.gpa);
            blocks.deinit(k.gpa);
        }
        try kv.split.planUnion(k.gpa, .{ .sel = sel, .k = kk, .table = table, .psh = a.psh, .world = k.world, .rank = k.rank, .row_bytes = block.Pool.row_bytes, .cap = k.opts.union_cap }, &blocks);
        if (blocks.items.len != 1) return k.keepBlocks(r, a.comp, a.send, a.recv, a.tok, kk, &blocks);
        const b = blocks.items[0];
        if (k.phys == null or k.phys.?.len < 4 * b.phys.len) {
            if (k.phys) |*p| p.free();
            k.phys = null;
            k.phys = try cuda.DeviceBuffer.alloc(r.d, @max(4 * b.phys.len, 256));
        }
        const up = (try pinned(r.d, &k.pin_up, 4 * (b.phys.len + b.tokens.len))).slice(u32);
        @memcpy(up[0..b.phys.len], b.phys);
        @memcpy(up[b.phys.len..][0..b.tokens.len], std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(b.tokens)));
        try cuda.DeviceBuffer.uploadAsync(k.phys.?, 0, std.mem.sliceAsBytes(up[0..b.phys.len]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = a.tok, .len = 4 * b.tokens.len }, 0, std.mem.sliceAsBytes(up[b.phys.len..][0..b.tokens.len]), r.stream.handle);
        try k.x.unionBlock(a.comp, block.Pool.row_bytes, k.phys.?.ptr, b.m, a.send, a.recv, r.stream.handle);
        k.stats.unions += 1;
        k.stats.union_rows_max = @max(k.stats.union_rows_max, @as(u64, b.m) * k.world);
        k.stats.bytes += @as(u64, b.m) * block.Pool.row_bytes * k.world;
    }

    /// The union by history (TF_DSV41_KV_SPLIT_HISTORY): no selection read back, no plan, no drain; M rows an owner.
    fn historyOf(k: *Kv, r: *run.Runner, a: anytype, n: usize, kk: u32, m: u32) !void {
        const h: kv.split.HistArgs = .{
            .sel = try r.tensorAddr(a.sel),
            .rows = @intCast(n),
            .k = kk,
            .table = a.table,
            .pages = k.slot.len,
            .psh = a.psh,
            .rank = k.rank,
            .world = k.world,
            .m = m,
            .row_bytes = block.Pool.row_bytes,
            .base = a.comp,
            .send = a.send,
            .tok = a.tok,
        };
        try k.x.history(h, a.recv, r.stream.handle);
        k.stats.unions += 1;
        k.stats.union_rows_max = @max(k.stats.union_rows_max, @as(u64, m) * k.world);
        k.stats.bytes += @as(u64, m) * block.Pool.row_bytes * k.world;
    }

    /// A pinned staging buffer of at least `len` bytes (grown, never shrunk).
    fn pinned(d: *const cuda.Driver, b: *?cuda.HostBuffer, len: usize) !cuda.HostBuffer {
        if (b.*) |x| if (x.bytes.len >= len) return x;
        if (b.*) |*x| x.free();
        b.* = null;
        b.* = try cuda.HostBuffer.alloc(d, @max(len, 256));
        return b.*.?;
    }

    fn dropBlocks(k: *Kv) void {
        for (k.blocks.items) |*b| b.deinit(k.gpa);
        k.blocks.clearRetainingCapacity();
    }

    /// A union of several row blocks: every block's rows and tokens uploaded once; exchanges wait for `fetchBlock`.
    fn keepBlocks(k: *Kv, r: *run.Runner, comp: u64, send: u64, recv: u64, tok: u64, kk: u32, blocks: *std.ArrayList(kv.split.Block)) !void {
        k.block_phys.clearRetainingCapacity();
        for (blocks.items) |b| {
            try k.block_phys.appendSlice(k.gpa, b.phys);
            try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = tok + 4 * @as(u64, b.a) * kk, .len = 4 * b.tokens.len }, 0, std.mem.sliceAsBytes(b.tokens));
        }
        const bytes = 4 * k.block_phys.items.len;
        if (k.phys == null or k.phys.?.len < bytes) {
            if (k.phys) |*p| p.free();
            k.phys = null;
            k.phys = try cuda.DeviceBuffer.alloc(r.d, @max(bytes, 256));
        }
        try cuda.DeviceBuffer.upload(k.phys.?, 0, std.mem.sliceAsBytes(k.block_phys.items));
        k.union_at = .{ .comp = comp, .send = send, .recv = recv };
        std.mem.swap(std.ArrayList(kv.split.Block), &k.blocks, blocks);
    }

    /// Block j of the kept union: its rows from their owners into the receive buffer (on the compute stream).
    pub fn fetchBlock(k: *Kv, r: *run.Runner, j: usize) !void {
        var off: u64 = 0;
        for (k.blocks.items[0..j]) |b| off += b.phys.len;
        const b = k.blocks.items[j];
        try k.x.unionBlock(k.union_at.comp, block.Pool.row_bytes, k.phys.?.ptr + 4 * off, b.m, k.union_at.send, k.union_at.recv, r.stream.handle);
        k.stats.unions += 1;
        k.stats.union_rows_max = @max(k.stats.union_rows_max, @as(u64, b.m) * k.world);
        k.stats.bytes += @as(u64, b.m) * block.Pool.row_bytes * k.world;
    }
};

/// kv/split.zig's device kernels over the engine's kvsplit ops (kernels_ops: the same argument layouts).
const Ops = struct {
    r: *run.Runner,

    fn kernels(o: *Ops) kv.split.Kernels {
        return .{ .ptr = o, .vtable = &.{ .dense = dense, .gather = gather, .pack = pack, .dedup = dedup, .history = history } };
    }

    fn self(ptr: *anyopaque) *Ops {
        return @ptrCast(@alignCast(ptr));
    }

    fn dense(ptr: *anyopaque, a: kv.split.DenseArgs, _: tp.collective.Stream) anyerror!void {
        const r = self(ptr).r;
        try r.kernels.others(r.stream).kvDense(.{ .sel = a.sel, .rows = a.rows, .k = a.k, .table = a.table, .pts = a.pts, .rslot = a.rslot, .psh = a.psh, .base = a.base, .row_bytes = a.row_bytes, .world = a.world, .send = a.send, .tok = a.tok });
    }

    fn gather(ptr: *anyopaque, base: u64, row_bytes: u32, phys: u64, n: u32, send: u64, _: tp.collective.Stream) anyerror!void {
        const r = self(ptr).r;
        try r.kernels.others(r.stream).kvGather(base, row_bytes, phys, n, send);
    }

    fn pack(ptr: *anyopaque, a: kv.split.PackArgs, _: tp.collective.Stream) anyerror!void {
        const r = self(ptr).r;
        try r.kernels.others(r.stream).kvPack(.{ .sel = a.sel, .rows = a.rows, .k = a.k, .table = a.table, .pts = a.pts, .rslot = a.rslot, .psh = a.psh, .rank = a.rank, .base = a.base, .row_bytes = a.row_bytes, .world = a.world, .send = a.send, .tok = a.tok, .lens = a.lens });
    }

    fn history(ptr: *anyopaque, a: kv.split.HistArgs, _: tp.collective.Stream) anyerror!void {
        const r = self(ptr).r;
        try r.kernels.others(r.stream).kvHistory(.{ .sel = a.sel, .rows = a.rows, .k = a.k, .table = a.table, .pages = a.pages, .psh = a.psh, .rank = a.rank, .world = a.world, .m = a.m, .row_bytes = a.row_bytes, .base = a.base, .send = a.send, .tok = a.tok });
    }

    fn dedup(ptr: *anyopaque, a: kv.split.DedupArgs, _: tp.collective.Stream) anyerror!void {
        const r = self(ptr).r;
        const p = a.p;
        try r.kernels.others(r.stream).kvDedup(.{
            .p = .{ .sel = p.sel, .rows = p.rows, .k = p.k, .table = p.table, .pts = p.pts, .rslot = p.rslot, .psh = p.psh, .rank = p.rank, .base = p.base, .row_bytes = p.row_bytes, .world = p.world, .send = p.send, .tok = p.tok, .lens = p.lens },
            .map = a.map,
            .first = a.first,
            .place = a.place,
            .bcnt = a.bcnt,
        });
    }
};
