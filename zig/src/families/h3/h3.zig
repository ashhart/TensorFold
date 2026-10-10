//! MiniMax H3 / FastH3's transformer on Metal: one denoising forward over the packed rows, int8 on the M5 tensor units.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const tensors = @import("tensors.zig");
const kernels = @import("h3_kernels");

pub const Tensors = tensors.Tensors;
pub const Ref = tensors.Ref;
pub const Checkpoint = tensors.Checkpoint;
pub const head_dim = 128;
pub const eps: f32 = 1e-5;
/// Rows of one routing tile, fixed by the checkpoint's training.
const slot_tile = 64;
/// Rows and columns one threadgroup of the int8 products owns.
const tile = 128;
const wide_group = 1024;
const product_threads = 256;
/// The six modulation tables of a block, in the order the reference splits them.
const shift_a = 0;
const scale_a = 1;
const gate_a = 2;
const shift_m = 3;
const scale_m = 4;
const gate_m = 5;

/// An int8 projection (K, N) and its scale per output channel.
const Quant = struct { w: mtl.Buffer, s: mtl.Buffer };

const Block = struct { q: Quant, k: Quant, v: Quant, compress: Quant, out: Quant, fc_gate: Quant, fc_value: Quant, fc2: Quant, norm1: Ref, norm2: Ref, norm_q: Ref, norm_k: Ref, tables: Ref };

/// Kernel shapes under test: `defines` goes ahead of the source; `tile_scales` and `weights` pick attention forms.
pub const Weights = enum { w8, w7, half };
pub const Options = struct { swiglu_rows: usize = 128, tile_scales: bool = true, weights: Weights = .w8, defines: []const u8 = "" };

pub const Timing = struct { wall: f64, gpu: f64 };

/// Where a profiled forward's GPU time went.
pub const stages = [_][]const u8{ "rows in", "norms", "q k v gate", "heads", "routing", "attention", "gate mix", "quantize", "attention out", "mlp in", "mlp out", "final" };

const Pass = struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder };

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
    /// Packed rows: `text`, then `condition` (a first frame's rows, held through the run), then `audio`, then `video`.
    rows: usize,
    text: usize,
    condition: usize,
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
    // the stream and a branch's rows, the projections at the attention width, the MLP's wide rows
    x: mtl.Buffer,
    y: mtl.Buffer,
    qr: mtl.Buffer,
    kr: mtl.Buffer,
    vr: mtl.Buffer,
    cr: mtl.Buffer,
    wide: mtl.Buffer,
    // int8 rows and their scales: the stream's width, the attention's, the MLP's
    q8: mtl.Buffer,
    xs: mtl.Buffer,
    a8: mtl.Buffer,
    as: mtl.Buffer,
    w8: mtl.Buffer,
    wxs: mtl.Buffer,
    // queries, keys and values in tile order, their scales, the tiles' means, scores, chosen tiles, pooled branch
    tq8: mtl.Buffer,
    tqs: mtl.Buffer,
    tk8: mtl.Buffer,
    tks: mtl.Buffer,
    tv8: mtl.Buffer,
    tvs: mtl.Buffer,
    qp: mtl.Buffer,
    kp: mtl.Buffer,
    vp: mtl.Buffer,
    tile_scores: mtl.Buffer,
    value_sums: mtl.Buffer,
    chosen: mtl.Buffer,
    coarse: mtl.Buffer,
    condition_latents: mtl.Buffer,
    video_latents: mtl.Buffer,
    audio_latents: mtl.Buffer,
    normed: mtl.Buffer,
    video_velocity: mtl.Buffer,
    audio_velocity: mtl.Buffer,
    /// Every query tile attends to every tile and the pooled branch is left out (not how the checkpoint was trained).
    dense: bool = false,
    /// One k and one v scale a key tile instead of one a key (the kernels' H3_TILE_SCALES).
    tile_scales: bool = false,
    weights: Weights = .w8,
    /// Rows one threadgroup of the SwiGLU's first product owns.
    swiglu_rows: usize = 128,
    profile: bool = false,
    spent: [stages.len]f64 = @splat(0),

    fn pipeline(self: *const Model, comptime name: []const u8) mtl.Pipeline {
        inline for (kernels.names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return self.pipelines[i];
        @compileError("no kernel " ++ name);
    }

    fn paddedRows(self: *const Model) usize {
        return (self.rows + tile - 1) / tile * tile;
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
        self.swiglu_rows = options.swiglu_rows;
        self.tile_scales = options.tile_scales;
        // the int8 weights read one value scale a key tile
        self.weights = if (options.tile_scales) options.weights else .half;
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

        // a case made for a first frame has a seventh count, the keyframe's rows
        const counts = (case.names.get("geometry") orelse return error.MissingTensor).shape[0];
        const geometry = try case.copy(i32, gpa, try case.at("geometry", .i32, counts), counts);
        defer gpa.free(geometry);
        self.prefix_tiles = @intCast(geometry[0]);
        self.tiles = @intCast(geometry[0] + geometry[1]);
        self.keep = @intCast(geometry[2]);
        self.text = @intCast(geometry[3]);
        self.audio = @intCast(geometry[4]);
        self.video = @intCast(geometry[5]);
        self.condition = if (counts > 6) @intCast(geometry[6]) else 0;
        self.rows = self.text + self.condition + self.audio + self.video;
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
        const source = try std.fmt.allocPrint(gpa, "#define H3P_HIDDEN {d}\n#define H3P_MLP {d}\n#define H3P_WIDE_GROUP {d}\n#define H3P_HEADS {d}\n#define H3P_ROWS {d}\n#define H3P_KEYS {d}\n#define H3_INNER {d}\n#define H3_SLOTS {d}\n#define H3_TILES {d}\n#define H3_SWT {d}\n{s}{s}{s}\n{s}", .{ hidden, self.mlp, wide_group, heads, rows, rows, inner, self.slots(), self.tiles, options.swiglu_rows, if (options.tile_scales) "#define H3_TILE_SCALES\n" else "", if (options.weights == .w7) "#define H3_W7\n" else "", options.defines, kernels.source });
        defer gpa.free(source);
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
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
        const padded = self.paddedRows();
        // a product reads its rows up to the next whole tile, so every such buffer has a spare one
        inline for (.{ "x", "y" }) |f| @field(self, f) = try device.buffer((rows + tile) * hidden * 2, shared);
        inline for (.{ "qr", "kr", "vr", "cr" }) |f| @field(self, f) = try device.buffer((rows + tile) * inner * 2, shared);
        self.wide = try device.buffer((rows + tile) * self.mlp * 2, shared);
        self.q8 = try device.buffer(padded * hidden, shared);
        self.xs = try device.buffer(padded * 4, shared);
        self.a8 = try device.buffer(padded * inner, shared);
        self.as = try device.buffer(padded * 4, shared);
        self.w8 = try device.buffer(padded * self.mlp, shared);
        self.wxs = try device.buffer(padded * (self.mlp / wide_group) * 4, shared);
        inline for (.{ "tq8", "tk8", "tv8" }) |f| {
            @field(self, f) = try device.buffer(heads * total * head_dim, shared);
            @memset(@field(self, f).slice(u8, heads * total * head_dim), 0);
        }
        inline for (.{ "tqs", "tks", "tvs" }) |f| {
            @field(self, f) = try device.buffer(heads * total * 4, shared);
            @memset(@field(self, f).slice(f32, heads * total), 0);
        }
        inline for (.{ "qp", "kp", "vp", "coarse" }) |f| @field(self, f) = try device.buffer(heads * self.tiles * head_dim * 4, shared);
        self.tile_scores = try device.buffer(heads * self.tiles * self.tiles * 4, shared);
        self.value_sums = try device.buffer(heads * self.tiles * head_dim * 4, shared);
        self.chosen = try device.buffer(heads * (self.tiles - self.prefix_tiles) * (self.prefix_tiles + self.keep) * 4, shared);
        self.condition_latents = try device.buffer(@max(self.condition, 1) * self.video_in * 4, shared);
        if (self.condition > 0) {
            const n = self.condition * self.video_in;
            const held_rows = try case.copy(f32, gpa, try case.at("condition", .f32, n), n);
            defer gpa.free(held_rows);
            @memcpy(self.condition_latents.slice(f32, n), held_rows);
        }
        self.video_latents = try device.buffer(self.video * self.video_in * 4, shared);
        self.audio_latents = try device.buffer(self.audio * self.audio_in * 4, shared);
        self.normed = try device.buffer(rows * hidden * 4, shared);
        self.video_velocity = try device.buffer(self.video * self.video_in * 4, shared);
        self.audio_velocity = try device.buffer(self.audio * self.audio_in * 4, shared);

        self.blocks = try gpa.alloc(Block, layers);
        for (self.blocks, 0..) |*b, i| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = self.queue.commandBuffer();
            const enc = cb.compute(.serial);
            const square = hidden * inner;
            b.q = try self.quantize(enc, try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_q.weight", .{i}), square), hidden, inner);
            b.k = try self.quantize(enc, try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_k.weight", .{i}), square), hidden, inner);
            b.v = try self.quantize(enc, try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_v.weight", .{i}), square), hidden, inner);
            b.compress = try self.quantize(enc, try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_gate_compress.weight", .{i}), square), hidden, inner);
            b.out = try self.quantize(enc, try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_out.0.weight", .{i}), square), inner, hidden);
            // the checkpoint stores the SwiGLU's first projection as [value; gate]
            const fc1 = try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.ff.net.0.proj.weight", .{i}), 2 * self.mlp * hidden);
            b.fc_value = try self.quantize(enc, fc1, hidden, self.mlp);
            b.fc_gate = try self.quantize(enc, .{ .buffer = fc1.buffer, .offset = fc1.offset + self.mlp * hidden * 2 }, hidden, self.mlp);
            b.fc2 = try self.quantize(enc, try ck.ref(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.ff.net.2.weight", .{i}), self.mlp * hidden), self.mlp, hidden);
            enc.end();
            cb.commit();
            cb.wait();
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

    /// `w` stored (n, k) in int8 (k, n) with a scale per output channel, written by the GPU.
    fn quantize(self: *const Model, enc: mtl.ComputeEncoder, w: Ref, k: usize, n: usize) !Quant {
        const q = Quant{ .w = try self.device.buffer(k * n, mtl.ResourceOptions.shared), .s = try self.device.buffer(n * 4, mtl.ResourceOptions.shared) };
        enc.setPipeline(self.pipeline("h3_quant_weight_t"));
        enc.setBuffer(w.buffer, w.offset, 0);
        enc.setValue([2]i32{ @intCast(k), @intCast(n) }, 1);
        enc.setBuffer(q.w, 0, 2);
        enc.setBuffer(q.s, 0, 3);
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
        return q;
    }

    fn lap(self: *Model, pass: *Pass, comptime stage: []const u8) void {
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

    /// The stream's modulated norm to int8; with `gate` it first adds the branch in `y` gated by that table part.
    fn norm(self: *const Model, enc: mtl.ComputeEncoder, weight: Ref, gates: ?Ref, gate: i32, tables: Ref, scale: i32, shift: i32) void {
        const g = gates orelse tables;
        enc.setPipeline(self.pipeline("h3_norm_q8"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(self.y, 0, 1);
        enc.setBuffer(weight.buffer, weight.offset, 2);
        enc.setBuffer(g.buffer, g.offset, 3);
        enc.setBuffer(tables.buffer, tables.offset, 4);
        enc.setBuffer(self.line, 0, 5);
        enc.setValue([6]i32{ @intCast(self.rows), @intCast(self.hidden), @intCast(self.lines), if (gates == null) -1 else gate, scale, shift }, 6);
        enc.setValue([1]f32{eps}, 7);
        enc.setBuffer(self.q8, 0, 8);
        enc.setBuffer(self.xs, 0, 9);
        enc.dispatchGroups(mtl.Size.of(self.paddedRows(), 1, 1), mtl.Size.of(32, 1, 1));
    }

    /// d = x w for int8 rows with one scale a row (or a scale per wide group) and an int8 projection to `n` outputs.
    fn product(self: *const Model, enc: mtl.ComputeEncoder, comptime kernel: []const u8, x: mtl.Buffer, scales: mtl.Buffer, w: Quant, d: mtl.Buffer, n: usize) void {
        enc.setPipeline(self.pipeline(kernel));
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(scales, 0, 1);
        enc.setBuffer(w.w, 0, 2);
        enc.setBuffer(w.s, 0, 3);
        enc.setBuffer(d, 0, 4);
        enc.dispatchGroups(mtl.Size.of(n / tile, (self.rows + tile - 1) / tile, 1), mtl.Size.of(product_threads, 1, 1));
    }

    fn quantRows(self: *const Model, enc: mtl.ComputeEncoder, x: mtl.Buffer, q: mtl.Buffer, scales: mtl.Buffer, width: usize, group: usize) void {
        enc.setPipeline(self.pipeline("h3p_quant_rows"));
        enc.setBuffer(x, 0, 0);
        enc.setValue([3]i32{ @intCast(self.rows), @intCast(width), @intCast(group) }, 1);
        enc.setBuffer(q, 0, 2);
        enc.setBuffer(scales, 0, 3);
        enc.dispatchGroups(mtl.Size.of(width / group, self.paddedRows(), 1), mtl.Size.of(32, 1, 1));
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

    /// q, k and v into tile order as int8; `mode` as the kernel's P[3].
    fn headsLayout(self: *const Model, enc: mtl.ComputeEncoder, b: *const Block, mode: i32) void {
        enc.setPipeline(self.pipeline("h3_heads_q8"));
        enc.setBuffer(self.qr, 0, 0);
        enc.setBuffer(self.kr, 0, 1);
        enc.setBuffer(self.vr, 0, 2);
        enc.setBuffer(b.norm_q.buffer, b.norm_q.offset, 3);
        enc.setBuffer(b.norm_k.buffer, b.norm_k.offset, 4);
        enc.setBuffer(self.cos.buffer, self.cos.offset, 5);
        enc.setBuffer(self.sin.buffer, self.sin.offset, 6);
        enc.setBuffer(self.slot.buffer, self.slot.offset, 7);
        enc.setValue([4]i32{ @intCast(self.rows), @intCast(self.heads), @intCast(self.rot), mode }, 8);
        enc.setValue([1]f32{eps}, 9);
        enc.setBuffer(self.tq8, 0, 10);
        enc.setBuffer(self.tqs, 0, 11);
        enc.setBuffer(self.tk8, 0, 12);
        enc.setBuffer(self.tks, 0, 13);
        enc.setBuffer(self.tv8, 0, 14);
        enc.setBuffer(self.tvs, 0, 15);
        enc.dispatchGroups(mtl.Size.of(self.rows, self.heads, 1), mtl.Size.of(32, 1, 1));
    }

    fn attend(self: *const Model, enc: mtl.ComputeEncoder, list: mtl.Buffer, queries: usize, keys: usize, first: usize, per_query: bool) void {
        enc.setPipeline(if (self.weights == .half) self.pipeline("h3_attention_tiles") else self.pipeline("h3_attention_w8"));
        if (self.weights != .half) enc.setBuffer(self.value_sums, 0, 12);
        enc.setBuffer(self.tq8, 0, 0);
        enc.setBuffer(self.tqs, 0, 1);
        enc.setBuffer(self.tk8, 0, 2);
        enc.setBuffer(self.tks, 0, 3);
        enc.setBuffer(self.tv8, 0, 4);
        enc.setBuffer(self.tvs, 0, 5);
        enc.setBuffer(list, 0, 6);
        enc.setBuffer(self.sizes.buffer, self.sizes.offset, 7);
        enc.setBuffer(self.row_of, 0, 8);
        enc.setValue([4]i32{ @intCast(queries), @intCast(keys), @intCast(first), @intFromBool(per_query) }, 9);
        enc.setValue([1]f32{1.0 / @sqrt(@as(f32, head_dim))}, 10);
        enc.setBuffer(self.qr, 0, 11);
        enc.dispatchGroups(mtl.Size.of(queries, self.heads, 1), mtl.Size.of(256, 1, 1));
    }

    /// The video and audio velocities for the latent rows at denoising step `step` of the case's schedule.
    pub fn forward(self: *Model, video: []const f32, audio: []const f32, step: usize, video_velocity: []f32, audio_velocity: []f32) !Timing {
        const started = mtl.clock.seconds();
        const rows = self.rows;
        const hidden = self.hidden;
        const heads = self.heads;
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
        // a first frame's rows enter like video rows on every pass and are never stepped
        const media = self.text + self.condition;
        if (self.condition > 0) self.rowsIn(enc.*, self.condition_latents, self.w_video_in, self.b_video_in, self.condition, self.video_in, self.text);
        self.rowsIn(enc.*, self.audio_latents, self.w_audio_in, self.b_audio_in, self.audio, self.audio_in, media);
        self.rowsIn(enc.*, self.video_latents, self.w_video_in, self.b_video_in, self.video, self.video_in, media + self.audio);
        self.lap(&pass, "rows in");
        const video_tiles = self.tiles - self.prefix_tiles;
        for (self.blocks, 0..) |*b, index| {
            // the block before left its MLP's rows in y: its gated add runs with this block's first norm
            if (index == 0) self.norm(enc.*, b.norm1, null, 0, b.tables, scale_a, shift_a) else self.norm(enc.*, b.norm1, self.blocks[index - 1].tables, gate_m, b.tables, scale_a, shift_a);
            self.lap(&pass, "norms");
            self.product(enc.*, "h3_i8_in", self.q8, self.xs, b.q, self.qr, self.inner);
            self.product(enc.*, "h3_i8_in", self.q8, self.xs, b.k, self.kr, self.inner);
            self.product(enc.*, "h3_i8_in", self.q8, self.xs, b.v, self.vr, self.inner);
            if (!self.dense) self.product(enc.*, "h3_i8_in", self.q8, self.xs, b.compress, self.cr, self.inner);
            self.lap(&pass, "q k v gate");
            if (self.tile_scales) {
                self.headsLayout(enc.*, b, 1);
                enc.barrier();
                enc.setPipeline(self.pipeline("h3_tile_scales"));
                enc.setBuffer(self.tks, 0, 0);
                enc.setBuffer(self.tvs, 0, 1);
                enc.setBuffer(self.sizes.buffer, self.sizes.offset, 2);
                enc.dispatchGroups(mtl.Size.of(self.tiles, heads, 1), mtl.Size.of(32, 1, 1));
                enc.barrier();
                self.headsLayout(enc.*, b, 2);
            } else self.headsLayout(enc.*, b, 0);
            if (self.weights == .w8) {
                enc.barrier();
                enc.setPipeline(self.pipeline("h3_value_sums"));
                enc.setBuffer(self.tv8, 0, 0);
                enc.setBuffer(self.value_sums, 0, 1);
                enc.dispatchGroups(mtl.Size.of(self.tiles, heads, 1), mtl.Size.of(32, 1, 1));
            }
            self.lap(&pass, "heads");
            if (self.dense) {
                self.attend(enc.*, self.all_tiles, self.tiles, self.tiles, 0, false);
                self.lap(&pass, "attention");
            } else {
                enc.setPipeline(self.pipeline("h3_pool"));
                enc.setBuffer(self.tq8, 0, 0);
                enc.setBuffer(self.tqs, 0, 1);
                enc.setBuffer(self.tk8, 0, 2);
                enc.setBuffer(self.tks, 0, 3);
                enc.setBuffer(self.tv8, 0, 4);
                enc.setBuffer(self.tvs, 0, 5);
                enc.setBuffer(self.sizes.buffer, self.sizes.offset, 6);
                enc.setBuffer(self.qp, 0, 7);
                enc.setBuffer(self.kp, 0, 8);
                enc.setBuffer(self.vp, 0, 9);
                enc.dispatchGroups(mtl.Size.of(self.tiles, heads, 1), mtl.Size.of(32, 1, 1));
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
                enc.dispatchGroups(mtl.Size.of(video_tiles, heads, 1), mtl.Size.of(32, 1, 1));
                enc.setPipeline(self.pipeline("h3_coarse"));
                enc.setBuffer(self.tile_scores, 0, 0);
                enc.setBuffer(self.vp, 0, 1);
                enc.setValue([1]i32{@intCast(self.tiles)}, 2);
                enc.setBuffer(self.coarse, 0, 3);
                enc.dispatchGroups(mtl.Size.of(self.tiles, heads, 1), mtl.Size.of(32, 1, 1));
                self.lap(&pass, "routing");
                // the prefix's query tiles see every tile; a video tile sees the prefix and its chosen tiles
                self.attend(enc.*, self.all_tiles, self.prefix_tiles, self.tiles, 0, false);
                self.attend(enc.*, self.chosen, video_tiles, self.prefix_tiles + self.keep, self.prefix_tiles, true);
                self.lap(&pass, "attention");
                // the pooled branch gated in and the rows rounded to int8, in one pass
                enc.setPipeline(self.pipeline("h3_mix_quant"));
                enc.setBuffer(self.qr, 0, 0);
                enc.setBuffer(self.cr, 0, 1);
                enc.setBuffer(self.coarse, 0, 2);
                enc.setBuffer(self.slot.buffer, self.slot.offset, 3);
                enc.setValue([3]i32{ @intCast(rows), @intCast(heads), @intCast(self.tiles) }, 4);
                enc.setBuffer(self.a8, 0, 5);
                enc.setBuffer(self.as, 0, 6);
                enc.dispatchGroups(mtl.Size.of(self.paddedRows(), 1, 1), mtl.Size.of(32, 1, 1));
                self.lap(&pass, "gate mix");
            }
            if (self.dense) {
                self.quantRows(enc.*, self.qr, self.a8, self.as, self.inner, self.inner);
                self.lap(&pass, "quantize");
            }
            self.product(enc.*, "h3_i8_out", self.a8, self.as, b.out, self.y, hidden);
            self.lap(&pass, "attention out");
            self.norm(enc.*, b.norm2, b.tables, gate_a, b.tables, scale_m, shift_m);
            self.lap(&pass, "norms");
            enc.setPipeline(if (self.swiglu_rows == tile) self.pipeline("h3p_i8_swiglu") else self.pipeline("h3_i8_swiglu"));
            enc.setBuffer(self.q8, 0, 0);
            enc.setBuffer(self.xs, 0, 1);
            enc.setBuffer(b.fc_gate.w, 0, 2);
            enc.setBuffer(b.fc_gate.s, 0, 3);
            enc.setBuffer(b.fc_value.w, 0, 4);
            enc.setBuffer(b.fc_value.s, 0, 5);
            enc.setBuffer(self.wide, 0, 6);
            enc.dispatchGroups(mtl.Size.of(self.mlp / tile, (rows + self.swiglu_rows - 1) / self.swiglu_rows, 1), mtl.Size.of(product_threads, 1, 1));
            self.lap(&pass, "mlp in");
            self.quantRows(enc.*, self.wide, self.w8, self.wxs, self.mlp, wide_group);
            self.lap(&pass, "quantize");
            self.product(enc.*, "h3p_i8_linear_wide", self.w8, self.wxs, b.fc2, self.y, hidden);
            self.lap(&pass, "mlp out");
        }
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
