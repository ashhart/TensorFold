//! A loaded Nemotron-H model: config, checkpoint buffers, kernel-layout weights and compiled pipelines.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const draft_ids = @import("draft_ids.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const learned = @import("learned.zig");
const kern = @import("kernels.zig");
const pk = @import("prefill_kernels.zig");
const frags = @import("../../core/frags.zig");
const row = @import("../../core/row_projection.zig");
const fwd = @import("forward.zig");
const prefill = @import("prefill.zig");
const prefill_w = @import("prefill_w.zig");
const simd_attention = @import("simd_attention.zig");

pub const Model = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: wts.Weights,
    kernels: kern.Kernels,
    prefill: pk.Kernels,
    draft_ids: []u32 = &.{},
    dir: [:0]u8 = undefined, // the folder the shards were read from, which kept lessons are written back into
    load_seconds: f64 = 0,
    compile_seconds: f64 = 0,

    /// Load `dir` (an MLX affine Nemotron 3.5 Lightning folder); `mtp` also reads mtp-4bit.safetensors when present.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, mtp: bool) !*Model {
        const started = mtl.clock.seconds();
        const m = try allocator.create(Model);
        errdefer allocator.destroy(m);
        m.allocator = allocator;
        m.dir = try allocator.dupeSentinel(u8, dir, 0);
        errdefer allocator.free(m.dir);
        m.device = try mtl.Device.init();
        m.queue = try m.device.queue();

        m.config = try cfg.Config.read(allocator, io, dir);
        try cfg.checkShapes(m.config);
        if (m.config.dt_min != 0 or !std.math.isInf(m.config.dt_max)) return error.UnsupportedTimeStepLimit;

        // compile the kernels while the weights load
        var compiled: anyerror!kern.Kernels = error.NotCompiled;
        var wide: anyerror!pk.Kernels = error.NotCompiled;
        var seconds: [2]f64 = .{ 0, 0 };
        const Compile = struct {
            fn run(comptime lib: type, out: anytype, a: std.mem.Allocator, d: mtl.Device, t: *f64) void {
                const t0 = mtl.clock.seconds();
                out.* = lib.load(a, d);
                t.* = mtl.clock.seconds() - t0;
            }
        };
        m.checkpoint = ckpt.Checkpoint.init(allocator);
        errdefer m.checkpoint.deinit();
        // until load returns, a kernel set that compiled is released with any failure
        errdefer if (compiled) |*k| k.deinit() else |_| {};
        errdefer if (wide) |*k| k.deinit() else |_| {};
        {
            const thread = try std.Thread.spawn(.{}, Compile.run, .{ kern, &compiled, allocator, m.device, &seconds[0] });
            defer thread.join();
            const prompt_thread = try std.Thread.spawn(.{}, Compile.run, .{ pk, &wide, allocator, m.device, &seconds[1] });
            defer prompt_thread.join();
            try m.loadWeights(io, dir, mtp);
        }
        errdefer {
            m.weights.deinit();
            allocator.free(m.draft_ids);
        }
        m.compile_seconds = @max(seconds[0], seconds[1]);
        // a matrix at another width than 4/64 reads through the core's row kernels on every chip
        if (compiled) |*k| {
            if (k.rows == null and m.config.widthsTag() != 0) k.rows = try row.Pipelines.load(m.device);
        } else |_| {}
        m.kernels = try compiled;
        m.prefill = try wide;
        if (m.config.widthsTag() != 0) try m.loadTiles();
        errdefer if (m.kernels.bits) |p| {
            p.deinit();
            allocator.destroy(p);
            prefill_w.deinit(allocator, m.kernels.gathers);
        };
        try frags.check(m.device, m.queue, allocator);
        if (!m.device.tensorUnits()) try simd_attention.check(m.device, m.queue);
        m.load_seconds = mtl.clock.seconds() - started;
        return m;
    }

    /// The checkpoint's shards (and the MTP head's file), then the weights in the kernels' layouts.
    fn loadWeights(m: *Model, io: std.Io, dir: []const u8, mtp: bool) !void {
        const allocator = m.allocator;
        const files = try shards.shardFiles(allocator, io, dir);
        defer shards.freeShardFiles(allocator, files);
        for (files) |file| try m.checkpoint.addFile(m.device, file, "");
        if (mtp) {
            const path = try std.fmt.allocPrintSentinel(allocator, "{s}/mtp-4bit.safetensors", .{dir}, 0);
            defer allocator.free(path);
            if (std.c.access(path, 0) == 0) try m.checkpoint.addFile(m.device, path, "mtp.");
        }
        const drafting = m.checkpoint.has("mtp.layers.0.eh_proj.weight");
        m.draft_ids = if (drafting) try draft_ids.load(allocator, m.config.vocab) else &.{};
        errdefer allocator.free(m.draft_ids);
        const residuals = try learned.split(allocator, m.device, &m.checkpoint, m.config);
        m.weights = wts.load(allocator, m.device, &m.checkpoint, m.config, if (drafting) m.draft_ids else null) catch |e| {
            learned.free(residuals);
            return e;
        };
        errdefer m.weights.deinit();
        try learned.attach(&m.weights, residuals);
    }

    /// Prompt chunks' kernels at widths but 4/64, built now: simd_qmm_bits for dense matrices, qmmw_nax for experts.
    fn loadTiles(m: *Model) !void {
        const p = try m.allocator.create(row.simd_bits.Pipelines);
        errdefer m.allocator.destroy(p);
        p.* = try row.simd_bits.Pipelines.load(m.device);
        errdefer p.deinit();
        var it = m.weights.linears.valueIterator();
        while (it.next()) |lin| if (lin.tiles == null) {
            for (lin.stack[0..lin.stacked]) |s| if (fwd.tiled(s.rows)) {
                _ = try row.simd_bits.call(p, s.rows.bits, s.rows.n, s.rows.k, prefill.lane_rows + 1, lin.n, false);
            };
            if (lin.rows) |r| if (fwd.tiled(r)) {
                _ = try row.simd_bits.call(p, r.bits, r.n, r.k, prefill.lane_rows + 1, r.n, false);
            };
        };
        var gathers: std.ArrayList(prefill_w.Gathers) = .empty;
        errdefer {
            for (gathers.items) |g| for (g.tiles) |t| t.deinit();
            gathers.deinit(m.allocator);
        }
        for (m.weights.layers[0..m.config.layers]) |l| if (l == .moe) if (l.moe.routed) |r| for (r) |x| {
            if (prefill_w.missing(gathers.items, x)) try gathers.append(m.allocator, try prefill_w.compile(m.allocator, m.device, x.one.bits, @intCast(x.one.group)));
        };
        m.kernels.gathers = try gathers.toOwnedSlice(m.allocator);
        m.kernels.bits = p;
    }

    pub fn deinit(self: *Model) void {
        if (self.kernels.bits) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }
        prefill_w.deinit(self.allocator, self.kernels.gathers);
        self.prefill.deinit();
        self.kernels.deinit();
        self.weights.deinit();
        self.checkpoint.deinit();
        self.allocator.free(self.draft_ids);
        self.allocator.free(self.dir);
        self.queue.deinit();
        self.device.deinit();
        self.allocator.destroy(self);
    }
};

test "a load that fails after its kernels compile releases them" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    device.deinit();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg.lightning });
    // an empty shard, as an interrupted download leaves one: the weights fail while both kernel sets compile
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "" });
    var path: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expectError(error.ShortRead, Model.load(std.testing.allocator, io, dir, false));
}
