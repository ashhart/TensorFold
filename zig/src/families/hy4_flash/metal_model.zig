//! The Metal target graph. Lane-owned cache buffers and retained slots are supplied for each layer.
const std = @import("std");
const mtl = @import("metal");
const source = @import("kernel_sources");
const aff = @import("row_projection");
const load = @import("loader.zig");
const weights = @import("model_weights.zig");
const dense = @import("metal_dense.zig");
const routed = @import("metal_routed.zig");
const memory = @import("metal_engram.zig");
const engram = @import("engram.zig");
const ihc = @import("ihc.zig");

pub const Ref = dense.Ref;

const Block = struct {
    attention: dense.Attention,
    feed: union(enum) { dense: dense.Feed, routed: routed.Layer },
    memory: ?memory.Layer,

    fn upload(a: std.mem.Allocator, device: mtl.Device, host: *load.Loaded, i: u32, use_tensors: bool) !Block {
        const config = host.config.value.text_config;
        var w = try weights.TargetLayer.read(a, host, i, .{ .rank = 0, .world = 1 }, 1);
        defer w.deinit(a);
        var attention = try dense.Attention.uploadForModel(a, device, &w, config, use_tensors);
        errdefer attention.deinit();
        var mem = if (w.memory) |*v| try memory.Layer.upload(a, device, v, config.hidden_size, config.engram_fp8_embedding) else null;
        errdefer if (mem) |*v| v.deinit();
        return .{
            .attention = attention,
            .feed = switch (w.feed) {
                .dense => .{ .dense = try dense.Feed.upload(a, device, &w) },
                .routed => .{ .routed = try routed.Layer.uploadForModel(a, device, &w, config, use_tensors) },
            },
            .memory = mem,
        };
    }

    fn deinit(self: *Block) void {
        if (self.memory) |*v| v.deinit();
        switch (self.feed) {
            inline else => |*v| v.deinit(),
        }
        self.attention.deinit();
    }
};

const Head = struct {
    owned: dense.Buffers,
    embedding: aff.Weights,
    vocab: aff.Weights,
    norm: mtl.Buffer,
    merge: dense.Merge,
    lookup: mtl.Pipeline,

    fn upload(a: std.mem.Allocator, device: mtl.Device, w: weights.Head) !Head {
        var b: dense.Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const lib = try mtl.Library.fromSource(device, source.embed_norm, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const name: [:0]const u8 = switch (w.embedding.bits) {
            4 => "tf_embed_b4_g64",
            8 => "tf_embed_b8_g64",
            else => return error.BadHy4EmbedBits,
        };
        var out: Head = .{
            .owned = undefined,
            .embedding = try b.matrix(w.embedding),
            .vocab = try b.matrix(w.lm_head),
            .norm = try b.upload(w.norm.bytes),
            .merge = try dense.Merge.upload(&b, w.merge),
            .lookup = try mtl.Pipeline.init(device, lib, name, false),
        };
        out.owned = b;
        return out;
    }

    fn deinit(self: *Head) void {
        self.lookup.deinit();
        self.owned.clear();
    }

    fn embed(self: Head, e: mtl.ComputeEncoder, ids: Ref, out: Ref, rows: u32) void {
        const w = self.embedding;
        e.setPipeline(self.lookup);
        e.setBuffer(ids.buffer, ids.offset, 0);
        e.setBuffer(w.w, w.w_off, 1);
        e.setBuffer(w.scales, w.s_off, 2);
        e.setBuffer(w.biases, w.b_off, 3);
        e.setValue(@as(i32, @intCast(w.k)), 4);
        e.setBuffer(out.buffer, out.offset, 5);
        e.dispatchThreads(mtl.Size.of(w.k / @as(usize, if (w.bits == 4) 2 else 1), rows, 1), mtl.Size.of(128, 1, 1));
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    host: load.Loaded,
    blocks: []Block,
    head: Head,
    common: dense.Kernels,
    routed: routed.Kernels,
    memory: memory.Kernels,

    pub fn open(a: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Model {
        var host = try load.Loaded.open(a, io, dir);
        errdefer host.close();
        const t = host.config.value.text_config;
        var common = try dense.Kernels.loadForModel(a, device, t, host.bits);
        errdefer common.deinit();
        var route = try routed.Kernels.loadForModel(a, device, t, host.bits);
        errdefer route.deinit();
        var mem = try memory.Kernels.load(device);
        errdefer mem.deinit();
        var head = try Head.upload(a, device, try weights.Head.read(&host, 1));
        errdefer head.deinit();
        const blocks = try a.alloc(Block, t.num_hidden_layers);
        errdefer a.free(blocks);
        var built: usize = 0;
        errdefer for (blocks[0..built]) |*block| block.deinit();
        for (blocks, 0..) |*block, i| {
            block.* = try Block.upload(a, device, &host, @intCast(i), common.tensors != null);
            built += 1;
        }
        host.checkpoint.release();
        return .{
            .allocator = a,
            .host = host,
            .blocks = blocks,
            .head = head,
            .common = common,
            .routed = route,
            .memory = mem,
        };
    }

    pub fn deinit(self: *Model) void {
        for (self.blocks) |*block| block.deinit();
        self.allocator.free(self.blocks);
        self.head.deinit();
        self.memory.deinit();
        self.routed.deinit();
        self.common.deinit();
        self.host.close();
        self.* = undefined;
    }

    pub fn encode(self: *Model, e: mtl.ComputeEncoder, frame: *Frame, cache: []const LayerCache) !Ref {
        if (frame.rows == 0 or cache.len != self.blocks.len) return error.BadHy4MetalFrame;
        var x = frame.streams[0];
        var y = frame.streams[1];
        self.embedRows(e, frame.ids, x, frame.rows * ihc.streams);
        e.barrier();
        var mem_ix: usize = 0;
        for (self.blocks, cache) |*block, lane| {
            if (block.memory) |*m| {
                try self.memory.forward(&self.common, e, m, &frame.memory[mem_ix], x, y, frame.rows);
                e.barrier();
                std.mem.swap(Ref, &x, &y);
                mem_ix += 1;
            }
            try frame.common.bind(frame.rows, block.attention.heads, lane.key_stride);
            try self.common.attend(e, &block.attention, &frame.common, lane.rows, x, y);
            e.barrier();
            std.mem.swap(Ref, &x, &y);
            switch (block.feed) {
                .dense => |*w| try self.common.feed(e, w, &frame.common, x, y),
                .routed => |*w| try self.routed.forward(&self.common, e, w, &frame.common, &frame.routed, x, y),
            }
            e.barrier();
            std.mem.swap(Ref, &x, &y);
        }
        _ = try self.common.collapse(e, x, self.head.merge, self.head.norm, &frame.common);
        try self.readout(e, frame.common.normalized, frame.logits, frame.rows);
        return frame.logits;
    }

    pub fn embedRows(self: *const Model, e: mtl.ComputeEncoder, ids: Ref, out: Ref, rows: u32) void {
        self.head.embed(e, ids, out, rows);
    }

    pub fn readout(self: *const Model, e: mtl.ComputeEncoder, x: Ref, out: Ref, rows: u32) !void {
        try self.common.project(e, self.head.vocab, x, out, rows);
    }
};

pub const LayerCache = struct { rows: dense.CacheRows, key_stride: u32 };

pub const Frame = struct {
    own: dense.Buffers,
    capacity: u32,
    rows: u32 = 0,
    streams: [2]Ref,
    ids: Ref,
    logits: Ref,
    common: dense.Scratch,
    routed: routed.Scratch,
    memory: []memory.Scratch,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, model: *const Model, rows: u32, key_stride: u32) !Frame {
        const t = model.host.config.value.text_config;
        if (rows == 0 or rows > routed.max_rows) return error.BadHy4MetalFrame;
        var needed: usize = 0;
        var largest = &model.blocks[0].attention;
        for (model.blocks) |*b| {
            needed += @intFromBool(b.memory != null);
            if (b.attention.heads > largest.heads) largest = &b.attention;
        }
        const memories = try a.alloc(memory.Scratch, needed);
        errdefer a.free(memories);
        var made: usize = 0;
        errdefer for (memories[0..made]) |*m| m.deinit();
        for (memories) |*m| {
            m.* = try memory.Scratch.init(a, device, rows, t.hidden_size);
            made += 1;
        }
        const inner = @max(t.moe_intermediate_size * t.n_shared_experts, t.intermediate_size);
        var common = try dense.Scratch.initForModel(a, device, largest, inner, key_stride, rows, model.common.tensors != null);
        errdefer common.deinit();
        var route = try routed.Scratch.initForModel(a, device, t, rows, model.routed.tensors != null);
        errdefer route.deinit();
        var b: dense.Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        var out: Frame = .{
            .own = undefined,
            .capacity = rows,
            .streams = .{ try b.rows(rows, ihc.streams * t.hidden_size), try b.rows(rows, ihc.streams * t.hidden_size) },
            .ids = .{ .buffer = try b.alloc(@as(usize, rows) * ihc.streams * @sizeOf(u32)) },
            .logits = try b.rows(rows, t.vocab_size),
            .common = common,
            .routed = route,
            .memory = memories,
        };
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Frame) void {
        self.common.deinit();
        self.routed.deinit();
        for (self.memory) |*m| m.deinit();
        self.own.a.free(self.memory);
        self.own.clear();
        self.* = undefined;
    }

    /// Call after this frame's previous command buffer completes. History windows are newest token first.
    pub fn stage(self: *Frame, model: *const Model, tokens: []const u32, windows: []const [engram.order]u32, media: []const u32) !void {
        self.rows = 0;
        if (tokens.len == 0 or tokens.len > self.capacity or tokens.len != windows.len or tokens.len != media.len)
            return error.BadHy4MetalFrame;
        const vocab = model.host.config.value.text_config.vocab_size;
        const ids = self.ids.buffer.slice(u32, tokens.len * ihc.streams);
        for (tokens, 0..) |token, row| {
            if (token >= vocab or windows[row][0] != token) return error.BadHy4Token;
            @memset(ids[row * ihc.streams ..][0..ihc.streams], token);
        }
        var i: usize = 0;
        for (model.blocks) |*b| if (b.memory) |*m| {
            try self.memory[i].stage(self.own.a, m, windows, media);
            i += 1;
        };
        self.rows = @intCast(tokens.len);
    }
};
