//! A round of mixed rows (several streams' windows and prompt chunks) and its activation buffers.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const kernels = @import("kernels.zig");
const State = @import("state.zig").State;
const Ref = kernels.Ref;

/// One stream's rows in a round: a drafted window (logged, rolled back by `kept`) or a prompt chunk (`commit`).
pub const Segment = struct { slot: u32, rows: u32, commit: bool };

const Buf = struct { name: []const u8, bytes: usize };

/// Activations for up to `rows_max` rows (32-row tiles for the MMA kernels); shared and untracked buffers.
pub const Scratch = struct {
    rows_max: u32,
    rows: u32 = 0,
    nseg: u32 = 0,
    ids: mtl.Buffer,
    kseg: mtl.Buffer,
    mrows: mtl.Buffer,
    prefix: mtl.Buffer,
    blocks: mtl.Buffer,
    xin: mtl.Buffer,
    delta: mtl.Buffer,
    proj: mtl.Buffer,
    fa: mtl.Buffer,
    f: mtl.Buffer,
    braw: mtl.Buffer,
    g2: mtl.Buffer,
    y: mtl.Buffer,
    qa: mtl.Buffer,
    qan: mtl.Buffer,
    q: mtl.Buffer,
    kv: mtl.Buffer,
    qlat: mtl.Buffer,
    partial: mtl.Buffer,
    olat: mtl.Buffer,
    rlogits: mtl.Buffer,
    tids: mtl.Buffer,
    tw: mtl.Buffer,
    slots: mtl.Buffer,
    nslots: mtl.Buffer,
    pairs: mtl.Buffer,
    where: mtl.Buffer,
    lat: mtl.Buffer,
    act: mtl.Buffer,
    xout: mtl.Buffer,
    ysum: mtl.Buffer,
    ynorm: mtl.Buffer,
    up: mtl.Buffer,
    sact: mtl.Buffer,
    sh: mtl.Buffer,
    dact: mtl.Buffer,
    logits: mtl.Buffer,
    tokens: mtl.Buffer,
    part: mtl.Buffer,

    pub fn init(device: mtl.Device, c: *const Config, rows_max: u32) !Scratch {
        const R: usize = (rows_max + 31) / 32 * 32;
        const H: usize = c.hidden;
        const W: usize = c.kdaWidth();
        const P = R * c.topk;
        const heads: usize = c.mla_heads;
        var s: Scratch = undefined;
        s.rows_max = rows_max;
        s.rows = 0;
        s.nseg = 0;
        const sizes = .{
            .{ "ids", R * 4 },                                                    .{ "kseg", R * @sizeOf(kernels.KdaSeg) },
            .{ "mrows", R * @sizeOf(kernels.MlaRow) },                            .{ "prefix", R * H * 2 },
            .{ "blocks", (c.layers / c.block + 2) * R * H * 2 },                  .{ "xin", R * H * 2 },
            .{ "delta", R * H * 2 },                                              .{ "proj", R * 3 * W * 2 },
            .{ "fa", R * c.kda_dim * 2 },                                         .{ "f", R * W * 2 },
            .{ "braw", R * c.kda_heads * 2 },                                     .{ "g2", R * @max(W, heads * c.v_dim) * 2 },
            .{ "y", R * @max(W, heads * c.v_dim) * 2 },                           .{ "qa", R * c.q_lora * 2 },
            .{ "qan", R * c.q_lora * 2 },                                         .{ "q", R * heads * c.qHead() * 2 },
            .{ "kv", R * (c.kv_lora + c.rope) * 2 },                              .{ "qlat", R * heads * c.kv_lora * 4 },
            .{ "partial", R * heads * kernels.mla_splits * (c.kv_lora + 2) * 4 }, .{ "olat", R * heads * c.kv_lora * 4 },
            .{ "rlogits", R * c.experts * 4 },                                    .{ "tids", P * 4 },
            .{ "tw", P * 4 },                                                     .{ "slots", c.experts * 3 * 4 },
            .{ "nslots", 16 },                                                    .{ "pairs", P * 4 },
            .{ "where", P * 4 },                                                  .{ "lat", R * c.latent * 2 },
            .{ "act", P * c.moe_inter * 2 },                                      .{ "xout", P * c.latent * 2 },
            .{ "ysum", R * c.latent * 2 },                                        .{ "ynorm", R * c.latent * 2 },
            .{ "up", R * H * 2 },                                                 .{ "sact", R * c.shared_inter * 2 },
            .{ "sh", R * H * 2 },                                                 .{ "dact", R * c.dense_inter * 2 },
            .{ "logits", R * c.vocab * 2 },                                       .{ "tokens", R * 4 },
            .{ "part", kernels.partFloats(rows_max) * 4 },
        };
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        inline for (sizes) |b| @field(s, b[0]) = try device.buffer(@max(b[1], 16), opts);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        const info = @typeInfo(Scratch).@"struct";
        inline for (info.field_names, info.field_types) |f, T| if (T == mtl.Buffer) @field(s, f).deinit();
    }

    /// Lay out a round: tokens in segment order, each segment's KDA record and each row's MLA slot and position.
    pub fn load(s: *Scratch, state: *const State, segs: []const Segment, tokens: []const u32) !void {
        var rows: u32 = 0;
        for (segs) |g| rows += g.rows;
        if (rows != tokens.len or rows > s.rows_max) return error.RoundTooLarge;
        @memcpy(s.ids.slice(u32, rows), tokens);
        const ks = s.kseg.slice(kernels.KdaSeg, segs.len);
        const mr = s.mrows.slice(kernels.MlaRow, rows);
        var first: u32 = 0;
        for (segs, 0..) |g, j| {
            const st = state.streams[g.slot];
            if (!g.commit and g.rows > state.limits.log_rows) return error.WindowTooLarge;
            if (st.pos + g.rows > state.limits.max_ctx) return error.ContextFull;
            ks[j] = .{ .first = first, .rows = g.rows, .kept = st.kept, .flags = if (g.commit) kernels.seg_commit else 0, .slot = g.slot };
            for (0..g.rows) |r| mr[first + r] = .{ .slot = g.slot, .pos = st.pos + @as(u32, @intCast(r)) };
            first += g.rows;
        }
        s.rows = rows;
        s.nseg = @intCast(segs.len);
    }

    pub fn ref(s: *const Scratch, comptime name: []const u8) Ref {
        return .{ .buf = @field(s, name) };
    }
};
