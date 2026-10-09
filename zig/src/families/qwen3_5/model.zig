//! A native Qwen3.5-family model: validated config and geometry, affine checkpoint views and Metal pipelines.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");
const pk = @import("../nemotron/prefill_kernels.zig");
const frags = @import("../../core/frags.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    /// MLX's prompt-chunk kernels (the NAX 4-bit qmm for prompt rows); null when they fail to compile or check here.
    prompt: ?pk.Kernels,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const config = try cfg.Config.read(gpa, io, dir);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = config;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ckpt.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        // the prompt kernels compile while the weights load
        var prompt: anyerror!pk.Kernels = error.NotCompiled;
        const Compile = struct {
            fn run(out: *anyerror!pk.Kernels, a: std.mem.Allocator, d: mtl.Device) void {
                out.* = pk.load(a, d);
            }
        };
        {
            const thread = std.Thread.spawn(.{}, Compile.run, .{ &prompt, gpa, m.device }) catch null;
            defer if (thread) |t| t.join() else Compile.run(&prompt, gpa, m.device);
            const files = try shards.shardFiles(gpa, io, dir);
            defer shards.freeShardFiles(gpa, files);
            for (files) |path| try m.checkpoint.addFileSelected(m.device, path, "", "language_model.");
            m.weights = try weights.load(gpa, m.device, &m.checkpoint, config.g);
        }
        errdefer m.weights.deinit();
        m.kernels = try kernels.load(gpa, m.device, config.g);
        errdefer m.kernels.deinit();
        m.prompt = prompt catch null;
        if (m.prompt) |*p| frags.check(m.device, m.queue, gpa) catch {
            p.deinit();
            m.prompt = null;
        };
        return m;
    }

    pub fn deinit(m: *Model) void {
        if (m.prompt) |*p| p.deinit();
        m.kernels.deinit();
        m.weights.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
