//! The Qwen3.5 / Qwen3.6 text model on the GPU: the Python TextModel's tensors as device buffers with their dtypes.

const std = @import("std");
const hip = @import("hip");
const config = @import("../../../weights/config.zig");
const table = @import("../../../weights/table.zig");
const host = @import("../../../weights/host.zig");
const checkpoint = @import("../../../weights/checkpoint.zig");
const slicing = @import("../../../weights/slicing.zig");

const quant = hip.quant;
const Tensor = table.Tensor;
pub const DType = table.DType;

/// One uploaded tensor: its device address and the element type and shape of the bytes there.
pub const Buf = quant.Buf;

/// A projection on the device in its format: the tensors as that format keeps them.
pub const Projection = quant.Device;

/// A layer's E + 1 experts, the shared one last; `fused` is gate then up per expert when `gated`.
pub const Experts = struct {
    fused: Projection,
    gated: bool,
    down: Projection,
    width: usize,
    dims: usize,
    count: usize,
};

/// Router rows [E + 1, D], the shared gate row last: bf16 as the Python holds them and `rows32` widened.
pub const Routed = struct { router: Buf, rows32: Buf, experts: Experts, top_k: usize, remap: ?Buf = null };

pub const DenseMlp = struct { gate: Projection, up: Projection, down: Projection };

pub const Mlp = union(enum) { dense: DenseMlp, routed: Routed };

pub const LinearLayer = struct {
    input_norm: Buf,
    post_norm: Buf,
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    conv: Buf,
    a_log: Buf,
    dt_bias: Buf,
    gnorm: Buf,
    out: Projection,
    mlp: Mlp,
};

pub const FullLayer = struct {
    input_norm: Buf,
    post_norm: Buf,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Buf,
    k_norm: Buf,
    mlp: Mlp,
};

pub const Layer = union(enum) { linear: LinearLayer, full: FullLayer };

pub const MtpHead = struct {
    fc_e_norm: Buf,
    fc_h_norm: Buf,
    fc_e: Projection,
    fc_h: Projection,
    q_norm: Buf,
    k_norm: Buf,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    final_norm: Buf,
    /// `null` ties with the main model's output head.
    head: ?Projection,
    input_norm: ?Buf,
    post_norm: ?Buf,
    mlp: ?Mlp,
    gated: bool,
};

/// Device memory of one model; every `Buf` of it is freed by `deinit`.
const Uploader = struct {
    up: hip.Upload,

    fn tensor(u: Uploader, t: Tensor) !Buf {
        return u.up.uploader().tensor(t);
    }

    fn projection(u: Uploader, p: host.Projection) !Projection {
        return quant.upload(u.up.uploader(), p);
    }

    fn mlp(u: Uploader, m: host.Mlp) !Mlp {
        switch (m) {
            .dense => |d| return .{ .dense = .{ .gate = try u.projection(d.gate), .up = try u.projection(d.up), .down = try u.projection(d.down) } },
            .routed => |r| {
                const e = r.experts;
                return .{ .routed = .{
                    .router = try u.tensor(r.router),
                    .rows32 = try u.tensor(r.rows32),
                    .experts = .{
                        .fused = try u.projection(e.fused),
                        .gated = e.gated,
                        .down = try u.projection(e.down),
                        .width = e.width,
                        .dims = e.dims,
                        .count = e.count,
                    },
                    .top_k = r.top_k,
                    .remap = if (r.remap) |t| try u.tensor(t) else null,
                } };
            },
        }
    }

    fn layer(u: Uploader, l: host.Layer) !Layer {
        switch (l.body) {
            .linear => |x| return .{ .linear = .{
                .input_norm = try u.tensor(x.input_norm),
                .post_norm = try u.tensor(x.post_norm),
                .qkv = try u.projection(x.qkv),
                .z = try u.projection(x.z),
                .a = try u.projection(x.a),
                .b = try u.projection(x.b),
                .conv = try u.tensor(x.conv),
                .a_log = try u.tensor(x.a_log),
                .dt_bias = try u.tensor(x.dt_bias),
                .gnorm = try u.tensor(x.gnorm),
                .out = try u.projection(x.out),
                .mlp = try u.mlp(x.mlp),
            } },
            .full => |x| return .{ .full = .{
                .input_norm = try u.tensor(x.input_norm),
                .post_norm = try u.tensor(x.post_norm),
                .q = try u.projection(x.q),
                .k = try u.projection(x.k),
                .v = try u.projection(x.v),
                .o = try u.projection(x.o),
                .q_norm = try u.tensor(x.q_norm),
                .k_norm = try u.tensor(x.k_norm),
                .mlp = try u.mlp(x.mlp),
            } },
        }
    }

    fn mtp(u: Uploader, m: host.Mtp) !MtpHead {
        return .{
            .fc_e_norm = try u.tensor(m.fc_e_norm),
            .fc_h_norm = try u.tensor(m.fc_h_norm),
            .fc_e = try u.projection(m.fc_e),
            .fc_h = try u.projection(m.fc_h),
            .q_norm = try u.tensor(m.q_norm),
            .k_norm = try u.tensor(m.k_norm),
            .q = try u.projection(m.q),
            .k = try u.projection(m.k),
            .v = try u.projection(m.v),
            .o = try u.projection(m.o),
            .final_norm = try u.tensor(m.final_norm),
            .head = if (m.head) |h| try u.projection(h) else null,
            .input_norm = if (m.input_norm) |n| try u.tensor(n) else null,
            .post_norm = if (m.post_norm) |n| try u.tensor(n) else null,
            .mlp = if (m.mlp) |x| try u.mlp(x) else null,
            .gated = m.gated,
        };
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    spec: config.Spec,
    buffers: std.ArrayList(hip.DeviceBuffer) = .empty,
    embed: Projection = undefined,
    layers: []Layer = &.{},
    final_norm: Buf = undefined,
    /// `null` ties the output head with `embed`.
    head: ?Projection = null,
    mtp: ?MtpHead = null,
    /// A tensor-parallel rank's share: the output projections hold fp32 shares of a sum.
    sliced: bool = false,

    /// The text tower of the checkpoint in `dir`, one layer at a time through host memory.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const hip.Runtime, dir: []const u8) !Model {
        return loadRank(gpa, io, d, dir, null);
    }

    /// As `load`, keeping only tensor-parallel rank `rank`'s share (null: the whole model).
    pub fn loadRank(gpa: std.mem.Allocator, io: std.Io, d: *const hip.Runtime, dir: []const u8, rank: ?slicing.Rank) !Model {
        var ck = try checkpoint.Checkpoint.open(gpa, io, dir);
        defer ck.close();
        return fromCheckpointRank(gpa, d, &ck, std.math.maxInt(usize), rank);
    }

    /// The first `limit` layers of `ck` (all when `limit` is large), with the MTP head only when every layer loads.
    pub fn fromCheckpoint(gpa: std.mem.Allocator, d: *const hip.Runtime, ck: *const checkpoint.Checkpoint, limit: usize) !Model {
        return fromCheckpointRank(gpa, d, ck, limit, null);
    }

    /// `fromCheckpoint` for one tensor-parallel rank, `spec` being what that rank sees.
    pub fn fromCheckpointRank(gpa: std.mem.Allocator, d: *const hip.Runtime, ck: *const checkpoint.Checkpoint, limit: usize, rank: ?slicing.Rank) !Model {
        const whole = ck.spec();
        var m: Model = .{ .gpa = gpa, .spec = if (rank) |r| try slicing.localSpec(whole, r) else whole, .sliced = rank != null };
        errdefer m.deinit();
        const u: Uploader = .{ .up = .{ .r = d, .buffers = &m.buffers, .gpa = gpa } };
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const embed = try ck.embed(scratch.allocator());
        m.embed = try u.projection(embed);
        m.final_norm = try u.tensor(try ck.finalNorm(scratch.allocator()));
        const head = try ck.head(scratch.allocator());
        if (rank) |r| {
            m.head = try u.projection(try slicing.vocabRows(scratch.allocator(), head orelse embed, whole.vocab, r));
        } else if (head) |h| m.head = try u.projection(h);
        var list: std.ArrayList(Layer) = .empty;
        errdefer list.deinit(gpa);
        for (0..@min(limit, m.spec.n_layers)) |i| {
            var host_layer = try ck.layer(i);
            defer host_layer.deinit();
            if (rank) |r| try slicing.layer(&host_layer, whole, r);
            try list.append(gpa, try u.layer(host_layer));
        }
        m.layers = try list.toOwnedSlice(gpa);
        // under tp only rank 0 drafts: its head is whole, logits rows included (its own or the model's whole head)
        const drafts = if (rank) |r| r.rank == 0 else true;
        if (drafts and m.layers.len == m.spec.n_layers) if (try ck.mtp()) |head_layer| {
            var h = head_layer;
            defer h.deinit();
            if (rank != null and h.head == null) h.head = head orelse embed;
            m.mtp = try u.mtp(h);
        };
        return m;
    }

    /// The output head: the embedding itself when the checkpoint ties them.
    pub fn outputHead(m: *const Model) Projection {
        return m.head orelse m.embed;
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};
