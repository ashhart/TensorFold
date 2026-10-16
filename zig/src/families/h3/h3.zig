//! MiniMax H3 / FastH3's transformer on Metal in the checkpoint's bf16: one denoising forward over the packed rows.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const sources = @import("kernel_sources");
const tensors = @import("tensors.zig");
const kernels = @import("h3_kernels");
const block = @import("block.zig");

pub const Tensors = tensors.Tensors;
pub const Ref = tensors.Ref;
pub const Checkpoint = tensors.Checkpoint;
pub const head_dim = 128;
pub const eps: f32 = 1e-5;
/// Rows of one routing tile, fixed by the checkpoint's training.
const slot_tile = 64;
/// The six modulation tables of a block, in the order the reference splits them.
pub const shift_a = 0;
pub const scale_a = 1;
pub const gate_a = 2;
pub const shift_m = 3;
pub const scale_m = 4;
pub const gate_m = 5;

/// A block's projections as the checkpoint stores them, (N, K) bf16 read in place; fc1 is [value; gate].
pub const Block = struct { q: Ref, k: Ref, v: Ref, compress: Ref, out: Ref, fc1: Ref, fc2: Ref, norm1: Ref, norm2: Ref, norm_q: Ref, norm_k: Ref, tables: Ref };

/// `defines` goes ahead of the kernels' source (a probe or a knock-out under test).
pub const Options = struct { defines: []const u8 = "" };

pub const Timing = struct { wall: f64, gpu: f64 };

/// Where a profiled forward's GPU time went.
pub const stages = [_][]const u8{ "rows in", "norms", "q k v gate", "heads", "routing", "attention", "gate mix", "quantize", "attention out", "mlp in", "mlp out", "final" };

pub const Pass = struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pipelines: [kernels.names.len]mtl.Pipeline,
    checkpoint: Checkpoint,
    case: *Tensors,
    hidden: usize,
    inner: usize,
    heads: usize,
    mlp: usize,
    video_in: usize,
    audio_in: usize,
    /// Packed rows: `text`, then `audio`, then `video`.
    rows: usize,
    text: usize,
    audio: usize,
    video: usize,
    tiles: usize,
    prefix_tiles: usize,
    keep: usize,
    lines: usize,
    rot: usize,
    blocks: []Block,
    prompt: []const u16,
    adaln: []const i32,
    times: []const i32,
    final_table: Ref,
    final_norm: Ref,
    w_video_in: Ref,
    b_video_in: Ref,
    w_audio_in: Ref,
    b_audio_in: Ref,
    w_video_out: Ref,
    b_video_out: Ref,
    w_audio_out: Ref,
    b_audio_out: Ref,
    cos: Ref,
    sin: Ref,
    slot: Ref,
    sizes: Ref,
    row_of: mtl.Buffer,
    all_tiles: mtl.Buffer,
    line: mtl.Buffer,
    time: mtl.Buffer,
    // the stream and a branch's rows, the normed rows, the projections at the attention width, the MLP's rows
    x: mtl.Buffer,
    y: mtl.Buffer,
    xn: mtl.Buffer,
    qr: mtl.Buffer,
    kr: mtl.Buffer,
    vr: mtl.Buffer,
    cr: mtl.Buffer,
    wide: mtl.Buffer,
    act: mtl.Buffer,
    // queries, keys and values in tile order, the tiles' means, scores, chosen tiles, pooled branch
    tq: mtl.Buffer,
    tk: mtl.Buffer,
    tv: mtl.Buffer,
    qp: mtl.Buffer,
    kp: mtl.Buffer,
    vp: mtl.Buffer,
    tile_scores: mtl.Buffer,
    chosen: mtl.Buffer,
    coarse: mtl.Buffer,
    video_latents: mtl.Buffer,
    audio_latents: mtl.Buffer,
    normed: mtl.Buffer,
    video_velocity: mtl.Buffer,
    audio_velocity: mtl.Buffer,
    /// Every query tile attends to every tile and the pooled branch is left out (not how the checkpoint was trained).
    dense: bool = false,
    profile: bool = false,
    spent: [stages.len]f64 = @splat(0),

    pub fn pipeline(self: *const Model, comptime name: []const u8) mtl.Pipeline {
        inline for (kernels.names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return self.pipelines[i];
        @compileError("no kernel " ++ name);
    }

    fn slots(self: *const Model) usize {
        return self.tiles * slot_tile;
    }

    /// `checkpoint_dir`: the FastH3 transformer folder of `shards` files. `case`: tools/zig/h3_case.py's output.
    pub fn load(gpa: std.mem.Allocator, checkpoint_dir: []const u8, shards: usize, case: *Tensors, options: Options) !*Model {
        const device = try mtl.Device.init();
        if (!device.tensorUnits()) return error.NeedsTensorUnits;
        const self = try gpa.create(Model);
        self.gpa = gpa;
        self.device = device;
        self.queue = try device.queue();
        self.case = case;
        self.dense = false;
        self.profile = false;
        self.spent = @splat(0);
        self.checkpoint = try Checkpoint.open(gpa, device, checkpoint_dir, shards);
        const ck = &self.checkpoint;
        const to_q = ck.entry("transformer_blocks.0.attn.to_q.weight") orelse return error.MissingTensor;
        const fc2 = ck.entry("transformer_blocks.0.ff.net.2.weight") orelse return error.MissingTensor;
        self.inner = to_q.shape[0];
        self.hidden = to_q.shape[1];
        self.heads = self.inner / head_dim;
        self.mlp = fc2.shape[1];
        var layers: usize = 0;
        var name: [128]u8 = undefined;
        while (ck.entry(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.norm1.weight", .{layers})) != null) layers += 1;

        const counts = (case.names.get("geometry") orelse return error.MissingTensor).shape[0];
        // a case made for a first frame has a seventh count, the keyframe's rows: not read here
        if (counts != 6) return error.UnsupportedCase;
        const geometry = try case.copy(i32, gpa, try case.at("geometry", .i32, counts), counts);
        defer gpa.free(geometry);
        self.prefix_tiles = @intCast(geometry[0]);
        self.tiles = @intCast(geometry[0] + geometry[1]);
        self.keep = @intCast(geometry[2]);
        self.text = @intCast(geometry[3]);
        self.audio = @intCast(geometry[4]);
        self.video = @intCast(geometry[5]);
        self.rows = self.text + self.audio + self.video;
        const rows = self.rows;
        const hidden = self.hidden;
        const inner = self.inner;
        const heads = self.heads;
        self.video_in = (case.names.get("video_in.weight") orelse return error.MissingTensor).shape[1];
        self.audio_in = (case.names.get("audio_in.weight") orelse return error.MissingTensor).shape[1];
        self.lines = (case.names.get("tables.0") orelse return error.MissingTensor).shape[1];
        self.rot = (case.names.get("cos") orelse return error.MissingTensor).shape[1];
        const times = (case.names.get("final") orelse return error.MissingTensor).shape[0];

        // the kernels take the model's sizes as compile-time values
        const source = try std.fmt.allocPrint(gpa, "#define H3_HEADS {d}\n#define H3_SLOTS {d}\n#define H3_TILES {d}\n{s}\n{s}", .{ heads, self.slots(), self.tiles, options.defines, kernels.source });
        defer gpa.free(source);
        // MLX's bf16 GEMM (the prefill kernels' gemm_nax) ahead of the family's own kernels; nax.h inlined
        const gemm = try std.mem.replaceOwned(u8, gpa, gemmSource(), "#include \"../nax.h\"", sources.nax);
        defer gpa.free(gemm);
        const all = try std.mem.concat(gpa, u8, &.{ gemm, source });
        defer gpa.free(all);
        const lib = try mtl.Library.fromSource(device, all, mtl.CompileOptions.mlx());
        defer lib.deinit();
        for (kernels.names, 0..) |kernel, i| self.pipelines[i] = try mtl.Pipeline.init(device, lib, kernel, false);

        self.prompt = try case.copy(u16, gpa, try case.at("text", .bf16, self.text * hidden), self.text * hidden);
        self.final_table = try self.held("final", .bf16, times * 2 * hidden);
        self.final_norm = try self.held("final_norm.weight", .bf16, hidden);
        self.w_video_in = try self.held("video_in.weight", .f32, hidden * self.video_in);
        self.b_video_in = try self.held("video_in.bias", .f32, hidden);
        self.w_audio_in = try self.held("audio_in.weight", .f32, hidden * self.audio_in);
        self.b_audio_in = try self.held("audio_in.bias", .f32, hidden);
        self.w_video_out = try self.held("video_out.weight", .f32, hidden * self.video_in);
        self.b_video_out = try self.held("video_out.bias", .f32, self.video_in);
        self.w_audio_out = try self.held("audio_out.weight", .f32, hidden * self.audio_in);
        self.b_audio_out = try self.held("audio_out.bias", .f32, self.audio_in);
        self.cos = try self.held("cos", .f32, rows * self.rot);
        self.sin = try self.held("sin", .f32, rows * self.rot);
        self.slot = try self.held("tile_slot", .i32, rows);
        self.sizes = try self.held("tile_sizes", .i32, self.tiles);

        const steps = (case.names.get("adaln") orelse return error.MissingTensor).shape[0];
        self.adaln = try case.copy(i32, gpa, try case.at("adaln", .i32, steps * rows), steps * rows);
        self.times = try case.copy(i32, gpa, try case.at("times", .i32, steps * rows), steps * rows);
        const total = self.slots();
        const shared = mtl.ResourceOptions.shared;
        self.row_of = try device.buffer(total * 4, shared);
        @memset(self.row_of.slice(i32, total), -1);
        for (self.slot.buffer.slice(i32, rows), 0..) |at, row| self.row_of.slice(i32, total)[@intCast(at)] = @intCast(row);
        self.all_tiles = try device.buffer(self.tiles * 4, shared);
        for (self.all_tiles.slice(i32, self.tiles), 0..) |*t, i| t.* = @intCast(i);
        self.line = try device.buffer(rows * 4, shared);
        self.time = try device.buffer(rows * 4, shared);
        inline for (.{ "x", "y", "xn" }) |f| @field(self, f) = try device.buffer(rows * hidden * 2, shared);
        inline for (.{ "qr", "kr", "vr", "cr" }) |f| @field(self, f) = try device.buffer(rows * inner * 2, shared);
        self.wide = try device.buffer(rows * 2 * self.mlp * 2, shared);
        self.act = try device.buffer(rows * self.mlp * 2, shared);
        // padding slots are never written and must read as zeros
        const slot_bytes = heads * total * head_dim * 2;
        inline for (.{ "tq", "tk", "tv" }) |f| {
            @field(self, f) = try device.buffer(slot_bytes, shared);
            @memset(@field(self, f).slice(u8, slot_bytes), 0);
        }
        inline for (.{ "qp", "kp", "vp", "coarse" }) |f| @field(self, f) = try device.buffer(heads * self.tiles * head_dim * 4, shared);
        self.tile_scores = try device.buffer(heads * self.tiles * self.tiles * 4, shared);
        self.chosen = try device.buffer(heads * (self.tiles - self.prefix_tiles) * (self.prefix_tiles + self.keep) * 4, shared);
        self.video_latents = try device.buffer(self.video * self.video_in * 4, shared);
        self.audio_latents = try device.buffer(self.audio * self.audio_in * 4, shared);
        self.normed = try device.buffer(rows * hidden * 2, shared);
        self.video_velocity = try device.buffer(self.video * self.video_in * 4, shared);
        self.audio_velocity = try device.buffer(self.audio * self.audio_in * 4, shared);

        self.blocks = try gpa.alloc(Block, layers);
        for (self.blocks, 0..) |*b, i| {
            const parts = .{ .{ "q", "attn.to_q", inner, hidden }, .{ "k", "attn.to_k", inner, hidden }, .{ "v", "attn.to_v", inner, hidden }, .{ "compress", "attn.to_gate_compress", inner, hidden }, .{ "out", "attn.to_out.0", hidden, inner }, .{ "fc1", "ff.net.0.proj", 2 * self.mlp, hidden }, .{ "fc2", "ff.net.2", hidden, self.mlp } };
            inline for (parts) |p| @field(b, p[0]) = try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}." ++ p[1] ++ ".weight", .{i}), p[2] * p[3]);
            b.norm1 = try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.norm1.weight", .{i}), hidden);
            b.norm2 = try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.norm2.weight", .{i}), hidden);
            b.norm_q = try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_q.weight", .{i}), head_dim);
            b.norm_k = try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_k.weight", .{i}), head_dim);
            b.tables = try self.held(try std.fmt.bufPrint(&name, "tables.{d}", .{i}), .bf16, 6 * self.lines * hidden);
        }
        return self;
    }

    /// The case's tensor `name` in a buffer of its own (the case file's tensors start wherever its header ends).
    fn held(self: *const Model, name: []const u8, dtype: st.DType, count: usize) !Ref {
        const from = try self.case.at(name, dtype, count);
        const bytes = count * dtype.size();
        const buffer = try self.device.buffer(bytes, mtl.ResourceOptions.shared);
        @memcpy(buffer.slice(u8, bytes), self.case.map.bytes[from.offset..][0..bytes]);
        return .{ .buffer = buffer };
    }

    pub fn lap(self: *Model, pass: *Pass, comptime stage: []const u8) void {
        pass.enc.barrier();
        if (!self.profile) return;
        pass.enc.end();
        pass.cb.commit();
        pass.cb.wait();
        inline for (stages, 0..) |s, i| if (comptime std.mem.eql(u8, s, stage)) {
            self.spent[i] += pass.cb.gpuSeconds();
        };
        pass.cb = self.queue.commandBuffer();
        pass.enc = pass.cb.compute(.concurrent);
    }

    /// After the tiles' means: tile scores, each video tile's chosen key tiles and the pooled branch.
    pub fn route(self: *const Model, enc: mtl.ComputeEncoder) void {
        const heads = self.heads;
        enc.barrier();
        enc.setPipeline(self.pipeline("h3_tile_scores"));
        enc.setBuffer(self.qp, 0, 0);
        enc.setBuffer(self.kp, 0, 1);
        enc.setValue([2]i32{ @intCast(self.tiles), @intCast(heads) }, 2);
        enc.setBuffer(self.tile_scores, 0, 3);
        enc.dispatchThreads(mtl.Size.of(self.tiles, self.tiles, heads), mtl.Size.of(32, 8, 1));
        enc.barrier();
        enc.setPipeline(self.pipeline("h3_topk"));
        enc.setBuffer(self.tile_scores, 0, 0);
        enc.setValue([3]i32{ @intCast(self.tiles), @intCast(self.prefix_tiles), @intCast(self.keep) }, 1);
        enc.setBuffer(self.chosen, 0, 2);
        enc.dispatchGroups(mtl.Size.of(self.tiles - self.prefix_tiles, heads, 1), mtl.Size.of(32, 1, 1));
        enc.setPipeline(self.pipeline("h3_coarse"));
        enc.setBuffer(self.tile_scores, 0, 0);
        enc.setBuffer(self.vp, 0, 1);
        enc.setValue([1]i32{@intCast(self.tiles)}, 2);
        enc.setBuffer(self.coarse, 0, 3);
        enc.dispatchGroups(mtl.Size.of(self.tiles, heads, 1), mtl.Size.of(32, 1, 1));
    }

    fn rowsIn(self: *const Model, enc: mtl.ComputeEncoder, from: mtl.Buffer, w: Ref, b: Ref, count: usize, width: usize, first: usize) void {
        enc.setPipeline(self.pipeline("h3_rows_in"));
        enc.setBuffer(from, 0, 0);
        enc.setBuffer(w.buffer, w.offset, 1);
        enc.setBuffer(b.buffer, b.offset, 2);
        enc.setValue([4]i32{ @intCast(count), @intCast(width), @intCast(self.hidden), @intCast(first) }, 3);
        enc.setBuffer(self.x, 0, 4);
        enc.dispatchThreads(mtl.Size.of(self.hidden, count, 1), mtl.Size.of(64, 16, 1));
    }

    fn rowsOut(self: *const Model, enc: mtl.ComputeEncoder, w: Ref, b: Ref, to: mtl.Buffer, count: usize, width: usize, first: usize) void {
        enc.setPipeline(self.pipeline("h3_rows_out"));
        enc.setBuffer(self.normed, 0, 0);
        enc.setBuffer(w.buffer, w.offset, 1);
        enc.setBuffer(b.buffer, b.offset, 2);
        enc.setValue([4]i32{ @intCast(count), @intCast(self.hidden), @intCast(width), @intCast(first) }, 3);
        enc.setBuffer(to, 0, 4);
        enc.dispatchGroups(mtl.Size.of(width, count, 1), mtl.Size.of(32, 1, 1));
    }

    /// The video and audio velocities for the latent rows at denoising step `step` of the case's schedule.
    pub fn forward(self: *Model, video: []const f32, audio: []const f32, step: usize, video_velocity: []f32, audio_velocity: []f32) !Timing {
        const started = mtl.clock.seconds();
        const rows = self.rows;
        const hidden = self.hidden;
        // each row's line in the blocks' tables and in the final layer's, for this step
        @memcpy(self.line.slice(i32, rows), self.adaln[step * rows ..][0..rows]);
        @memcpy(self.time.slice(i32, rows), self.times[step * rows ..][0..rows]);
        @memcpy(self.x.slice(u16, self.prompt.len), self.prompt);
        @memcpy(self.video_latents.slice(f32, video.len), video);
        @memcpy(self.audio_latents.slice(f32, audio.len), audio);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var pass: Pass = undefined;
        pass.cb = self.queue.commandBuffer();
        pass.enc = pass.cb.compute(.concurrent);
        var gpu: f64 = 0;
        const enc = &pass.enc;
        const media = self.text;
        self.rowsIn(enc.*, self.audio_latents, self.w_audio_in, self.b_audio_in, self.audio, self.audio_in, media);
        self.rowsIn(enc.*, self.video_latents, self.w_video_in, self.b_video_in, self.video, self.video_in, media + self.audio);
        self.lap(&pass, "rows in");
        for (0..self.blocks.len) |index| block.run(self, &pass, index);
        const last = self.blocks[self.blocks.len - 1].tables;
        enc.setPipeline(self.pipeline("h3_gate_add"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(self.y, 0, 1);
        enc.setBuffer(last.buffer, last.offset, 2);
        enc.setBuffer(self.line, 0, 3);
        enc.setValue([4]i32{ @intCast(rows), @intCast(hidden), @intCast(self.lines), gate_m }, 4);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        enc.barrier();
        enc.setPipeline(self.pipeline("h3_final_norm"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(self.final_norm.buffer, self.final_norm.offset, 1);
        enc.setBuffer(self.final_table.buffer, self.final_table.offset, 2);
        enc.setBuffer(self.time, 0, 3);
        enc.setValue([2]i32{ @intCast(rows), @intCast(hidden) }, 4);
        enc.setValue([1]f32{eps}, 5);
        enc.setBuffer(self.normed, 0, 6);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        enc.barrier();
        self.rowsOut(enc.*, self.w_audio_out, self.b_audio_out, self.audio_velocity, self.audio, self.audio_in, media);
        self.rowsOut(enc.*, self.w_video_out, self.b_video_out, self.video_velocity, self.video, self.video_in, media + self.audio);
        pass.enc.end();
        pass.cb.commit();
        pass.cb.wait();
        gpu += pass.cb.gpuSeconds();
        if (self.profile) self.spent[stages.len - 1] += pass.cb.gpuSeconds();
        if (pass.cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        @memcpy(video_velocity, self.video_velocity.slice(f32, video_velocity.len));
        @memcpy(audio_velocity, self.audio_velocity.slice(f32, audio_velocity.len));
        return .{ .wall = mtl.clock.seconds() - started, .gpu = gpu };
    }
};

/// The prefill kernels' gemm_nax source, which the block's products use.
fn gemmSource() []const u8 {
    inline for (sources.prefill) |f| if (comptime std.mem.eql(u8, f.name, "gemm_nax")) return f.text;
    @compileError("no gemm_nax source");
}
