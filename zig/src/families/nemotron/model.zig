//! A loaded Nemotron-H model: config, checkpoint buffers, kernel-layout weights and compiled pipelines.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const draft_ids = @import("draft_ids.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const pk = @import("prefill_kernels.zig");
const frags = @import("../../core/frags.zig");
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
    load_seconds: f64 = 0,
    compile_seconds: f64 = 0,

    /// Load `dir` (an MLX 4-bit Nemotron 3.5 Lightning folder); `mtp` also reads mtp-4bit.safetensors when present.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, mtp: bool) !*Model {
        const started = mtl.clock.seconds();
        const m = try allocator.create(Model);
        errdefer allocator.destroy(m);
        m.allocator = allocator;
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
        m.kernels = try compiled;
        errdefer m.kernels.deinit();
        m.prefill = try wide;
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
        m.weights = try wts.load(allocator, m.device, &m.checkpoint, m.config, if (drafting) m.draft_ids else null);
    }

    pub fn deinit(self: *Model) void {
        self.prefill.deinit();
        self.kernels.deinit();
        self.weights.deinit();
        self.checkpoint.deinit();
        self.allocator.free(self.draft_ids);
        self.queue.deinit();
        self.device.deinit();
        self.allocator.destroy(self);
    }
};
