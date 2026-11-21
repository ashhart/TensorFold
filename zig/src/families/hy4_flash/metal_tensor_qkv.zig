//! Hy4-flash Q/KV projections and gate on Apple GPU tensor units.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const lane = @import("../../core/lane_projection.zig");
const source = @import("kernel_sources").hy4_flash_tensor_dispatch;
const config = @import("config.zig");
const mla = @import("mla_weights.zig");

pub const Packed = struct {
    w: lane.Ref,
    sb: lane.Ref,

    pub fn upload(owner: anytype, inputs: []const core.quant.mlx.Host, l: lane.Layout) !Packed {
        const words = try owner.a.alloc(u32, l.sourceWords());
        defer owner.a.free(words);
        const scales = try owner.a.alloc(u16, l.metadataElements() / 2);
        defer owner.a.free(scales);
        const bias = try owner.a.alloc(u16, scales.len);
        defer owner.a.free(bias);
        var wi: usize = 0;
        var si: usize = 0;
        for (inputs) |input| {
            if (input.bits != l.format.bits or input.group != l.format.group or input.scales.shape[1] * input.group != l.k or
                input.words.bytes.len % @sizeOf(u32) != 0 or input.scales.bytes.len % @sizeOf(u16) != 0)
                return error.BadQkvInput;
            const nw = input.words.bytes.len / 4;
            const nt = input.scales.bytes.len / 2;
            if (nw > words.len - wi or nt > scales.len - si or input.biases.bytes.len != input.scales.bytes.len)
                return error.BadQkvInput;
            @memcpy(std.mem.sliceAsBytes(words[wi..][0..nw]), input.words.bytes);
            @memcpy(std.mem.sliceAsBytes(scales[si..][0..nt]), input.scales.bytes);
            @memcpy(std.mem.sliceAsBytes(bias[si..][0..nt]), input.biases.bytes);
            wi += nw;
            si += nt;
        }
        if (wi != words.len or si != scales.len) return error.BadQkvInput;
        const w = try owner.alloc(l.weightWords() * 4);
        const sb = try owner.alloc(l.metadataElements() * 2);
        try lane.pack(l, words, scales, bias, w.slice(u32, l.weightWords()), sb.slice(u16, l.metadataElements()));
        if (l.reg and l.format.bits == 8) try lane.regOrder8(w.slice(u32, l.weightWords()), l.n, l.k) else if (l.reg) try lane.regOrder(owner.d, &.{.{ .w = .{ .buf = w }, .n = l.n, .k = l.k }});
        return .{ .w = .{ .buf = w }, .sb = .{ .buf = sb } };
    }
};

/// Rows one prompt chunk may hold (the routed layer's prompt_rows).
pub const prompt_rows = 512;

fn layout(bits: u8, n: usize, k: usize, sk: usize) lane.Layout {
    return .{
        .n = n,
        .k = k,
        .sk = sk,
        .format = .{ .bits = bits, .group = 64 },
        .reg = true,
        .precompute_sums = true,
        // a whole prompt chunk in one encode on the reg path (independent 16-row blocks, so the same bits)
        .rows = prompt_rows,
        .row_blocks = false,
    };
}

/// Any other dense projection [n, k] on tensor units
pub fn denseLayout(bits: u8, n: usize, k: usize) lane.Layout {
    return layout(bits, n, k, if (k < 2048) 2 else if (k <= 4096) 4 else 8);
}

pub fn inputLayout(t: config.Text, bits: u8, heads: u32) lane.Layout {
    return layout(bits, t.q_lora_rank + mla.absorbed_stride + heads * mla.value_width, t.hidden_size, 4);
}

pub fn queryLayout(t: config.Text, bits: u8, heads: u32) lane.Layout {
    return layout(bits, heads * mla.query_stride, t.q_lora_rank, 2);
}

pub fn flavor(t: config.Text, heads: u32) usize {
    return if (heads == t.swa_num_attention_heads) 1 else 0;
}

/// Passes of at most this many rows take the reg kernels that add their own group sums
pub const inline_rows = 8;

/// `l` with the kernel adding its split's group sums up front (4-bit reg and 8-bit layouts; null otherwise).
pub fn withInlineSums(l: lane.Layout) ?lane.Layout {
    if (!l.reg and l.format.bits != 8) return null;
    var out = l;
    out.inline_sums = true;
    return out;
}

pub const Kernels = struct {
    inputs: [2]lane.Projection,
    queries: [2]lane.Projection,
    /// The same projections adding their own group sums (4-bit reg layouts), for passes of inline_rows at most.
    inline_inputs: [2]?lane.Projection = .{ null, null },
    inline_queries: [2]?lane.Projection = .{ null, null },
    unpack: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, d: mtl.Device, t: config.Text, bits: u8) !Kernels {
        const lib = try mtl.Library.fromSource(d, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var unpack = try mtl.Pipeline.init(d, lib, "tf_hy4_qkv_split", false);
        errdefer unpack.deinit();
        var out: Kernels = .{ .inputs = undefined, .queries = undefined, .unpack = unpack };
        var made: usize = 0;
        errdefer {
            for (0..made) |i| {
                out.inputs[i].deinit();
                out.queries[i].deinit();
            }
        }
        for ([_]u32{ t.num_attention_heads, t.swa_num_attention_heads }, 0..) |heads, i| {
            var input = try lane.Projection.init(a, d, inputLayout(t, bits, heads));
            errdefer input.deinit();
            out.queries[i] = try lane.Projection.init(a, d, queryLayout(t, bits, heads));
            out.inputs[i] = input;
            made += 1;
        }
        for ([_]u32{ t.num_attention_heads, t.swa_num_attention_heads }, 0..) |heads, i| {
            if (withInlineSums(inputLayout(t, bits, heads))) |l| out.inline_inputs[i] = try lane.Projection.init(a, d, l);
            if (withInlineSums(queryLayout(t, bits, heads))) |l| out.inline_queries[i] = try lane.Projection.init(a, d, l);
        }
        return out;
    }

    pub fn deinit(self: Kernels) void {
        for (self.inputs, self.queries) |a, b| {
            a.deinit();
            b.deinit();
        }
        for (self.inline_inputs, self.inline_queries) |a, b| {
            if (a) |x| x.deinit();
            if (b) |x| x.deinit();
        }
        self.unpack.deinit();
    }

    pub fn inputFor(self: *const Kernels, which: usize, rows: u32) *const lane.Projection {
        if (rows <= inline_rows) if (self.inline_inputs[which]) |*p| return p;
        return &self.inputs[which];
    }

    pub fn queryFor(self: *const Kernels, which: usize, rows: u32) *const lane.Projection {
        if (rows <= inline_rows) if (self.inline_queries[which]) |*p| return p;
        return &self.queries[which];
    }

    pub fn split(self: Kernels, e: mtl.ComputeEncoder, input: lane.Ref, q: lane.Ref, kv: lane.Ref, gates: lane.Ref, rows: u32, q_rank: u32, gate_cols: u32) void {
        e.setPipeline(self.unpack);
        for ([_]lane.Ref{ input, q, kv, gates }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        // Metal's uint3 has the same 16-byte ABI as uint4.
        e.setValue([4]u32{ q_rank, mla.absorbed_stride, gate_cols, 0 }, 4);
        e.dispatchThreads(mtl.Size.of(q_rank + mla.absorbed_stride + gate_cols, rows, 1), mtl.Size.of(128, 1, 1));
    }
};
