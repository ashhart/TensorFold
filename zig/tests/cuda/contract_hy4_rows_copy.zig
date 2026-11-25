//! Byte-copy contract cases for 6 `hy4_rows` kernels (`slice_*`, `interleave2`, `splice_streams`, ViT helpers).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn randBf16(rng: std.Random) u16 {
    return c.toBf16(rng.float(f32) * 2.0 - 1.0);
}

pub const CopyKind = enum {
    slice_bf16,
    interleave2_bf16,
    slice_f32,
    splice_streams,
    vit_merge_gather,
    vit_append_newline,
};

const copy_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 40, .k = 64 },
    .{ .rows = 5, .n = 40, .k = 64 },
    .{ .rows = 17, .n = 40, .k = 64 },
};
const merge_shapes = [_]c.Shape{
    .{ .rows = 2, .n = 2, .k = 32 },
    .{ .rows = 4, .n = 6, .k = 32 },
    .{ .rows = 6, .n = 4, .k = 32 },
};
const newline_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 2, .k = 32 },
    .{ .rows = 3, .n = 5, .k = 32 },
    .{ .rows = 5, .n = 3, .k = 32 },
};

pub const CopyCase = struct {
    rig: rig_.Rig,
    rows: *const hr.Rows,
    kind: CopyKind,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    dev_a: u64 = 0,
    dev_b: u64 = 0,
    dev_out: u64 = 0,
    out_bytes: usize = 0,
    want: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *CopyCase) void {
        self.want.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *CopyCase) g.Kernel {
        const name: []const u8 = switch (self.kind) {
            .slice_bf16 => "cuda hy4_rows hy4_rows_slice_bf16",
            .interleave2_bf16 => "cuda hy4_rows hy4_rows_interleave2_bf16",
            .slice_f32 => "cuda hy4_rows hy4_rows_slice_f32",
            .splice_streams => "cuda hy4_rows hy4_rows_splice_streams",
            .vit_merge_gather => "cuda hy4_rows hy4_rows_vit_merge_gather",
            .vit_append_newline => "cuda hy4_rows hy4_rows_vit_append_newline",
        };
        const served: []const c.Shape = switch (self.kind) {
            .vit_merge_gather => &merge_shapes,
            .vit_append_newline => &newline_shapes,
            else => &copy_shapes,
        };
        return .{
            .spec = .{
                .name = name,
                .accum = .none,
                .depth = zeroDepth,
                .out = .raw,
                .replaces = "host exact byte copy",
                .nan = .{ .stated = "preserves bit patterns verbatim including NaN/Inf payloads" },
                .served = served,
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(CopyCase);

    pub fn prepare(self: *CopyCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ @backingInt(self.kind));
        const rng = prng.random();
        switch (self.kind) {
            .slice_bf16 => {
                const src = try a.alloc(u16, shape.rows * 64);
                defer a.free(src);
                for (src) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) src[0] = 0x7fc1;
                try self.want.resize(a, shape.rows * 40 * 2);
                const w16 = std.mem.bytesAsSlice(u16, self.want.items);
                for (0..shape.rows) |r| @memcpy(w16[r * 40 ..][0..40], src[r * 64 ..][0..40]);
                self.dev_a = try self.rig.input(std.mem.sliceAsBytes(src));
                self.out_bytes = self.want.items.len;
            },
            .interleave2_bf16 => {
                const half = 32;
                const src = try a.alloc(u16, 2 * shape.rows * half);
                defer a.free(src);
                for (src) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) src[1] = 0x7f80;
                try self.want.resize(a, shape.rows * 2 * half * 2);
                const w16 = std.mem.bytesAsSlice(u16, self.want.items);
                for (0..shape.rows) |r| {
                    @memcpy(w16[r * 2 * half ..][0..half], src[r * half ..][0..half]);
                    @memcpy(w16[r * 2 * half + half ..][0..half], src[shape.rows * half + r * half ..][0..half]);
                }
                self.dev_a = try self.rig.input(std.mem.sliceAsBytes(src));
                self.out_bytes = self.want.items.len;
            },
            .slice_f32 => {
                const src = try a.alloc(f32, shape.rows * 48);
                defer a.free(src);
                for (src) |*v| v.* = rng.float(f32) * 4.0 - 2.0;
                if (fill == .nonfinite) src[12] = std.math.nan(f32);
                try self.want.resize(a, shape.rows * 24 * 4);
                const wf = std.mem.bytesAsSlice(f32, self.want.items);
                for (0..shape.rows) |r| @memcpy(wf[r * 24 ..][0..24], src[r * 48 + 12 ..][0..24]);
                self.dev_a = try self.rig.input(std.mem.sliceAsBytes(src));
                self.out_bytes = self.want.items.len;
            },
            .splice_streams => {
                const hidden = 64;
                const n_feat = shape.rows + 2;
                const feat = try a.alloc(u16, n_feat * hidden);
                defer a.free(feat);
                for (feat) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) feat[0] = 0x7fc0;
                const idx = try a.alloc(i32, shape.rows);
                defer a.free(idx);
                for (0..shape.rows) |r| idx[r] = if ((r & 1) == 0) @intCast((r * 3) % n_feat) else -1;
                try self.want.resize(a, shape.rows * 4 * hidden * 2);
                @memset(self.want.items, 0x5a);
                const w16 = std.mem.bytesAsSlice(u16, self.want.items);
                for (0..shape.rows) |r| {
                    if (idx[r] < 0) continue;
                    const s: usize = @intCast(idx[r]);
                    for (0..4) |st| @memcpy(w16[(r * 4 + st) * hidden ..][0..hidden], feat[s * hidden ..][0..hidden]);
                }
                self.dev_a = try self.rig.input(std.mem.sliceAsBytes(feat));
                self.dev_b = try self.rig.input(std.mem.sliceAsBytes(idx));
                self.out_bytes = self.want.items.len;
            },
            .vit_merge_gather => {
                const gh = shape.rows;
                const gw = shape.n;
                const d = shape.k;
                const src = try a.alloc(u16, gh * gw * d);
                defer a.free(src);
                for (src) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) src[0] = 0x7fc0;
                const mh = gh / 2;
                const mw = gw / 2;
                try self.want.resize(a, mh * mw * 4 * d * 2);
                const w16 = std.mem.bytesAsSlice(u16, self.want.items);
                for (0..mh) |oh| for (0..mw) |ow| for (0..4) |sub| {
                    const sr = (oh * 2 + (sub >> 1)) * gw + (ow * 2 + (sub & 1));
                    const dr = (oh * mw + ow) * 4 + sub;
                    @memcpy(w16[dr * d ..][0..d], src[sr * d ..][0..d]);
                };
                self.dev_a = try self.rig.input(std.mem.sliceAsBytes(src));
                self.out_bytes = self.want.items.len;
            },
            .vit_append_newline => {
                const mh = shape.rows;
                const mw = shape.n;
                const d = shape.k;
                const merged = try a.alloc(u16, mh * mw * d);
                defer a.free(merged);
                const nl = try a.alloc(u16, d);
                defer a.free(nl);
                for (merged) |*v| v.* = randBf16(rng);
                for (nl) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) nl[0] = 0xff80;
                try self.want.resize(a, mh * (mw + 1) * d * 2);
                const w16 = std.mem.bytesAsSlice(u16, self.want.items);
                for (0..mh) |oh| {
                    for (0..mw) |ow| @memcpy(w16[(oh * (mw + 1) + ow) * d ..][0..d], merged[(oh * mw + ow) * d ..][0..d]);
                    @memcpy(w16[(oh * (mw + 1) + mw) * d ..][0..d], nl);
                }
                self.dev_a = try self.rig.input(std.mem.sliceAsBytes(merged));
                self.dev_b = try self.rig.input(std.mem.sliceAsBytes(nl));
                self.out_bytes = self.want.items.len;
            },
        }
        self.dev_out = try self.rig.output(self.out_bytes);
        return true;
    }

    pub fn run(self: *CopyCase) !void {
        switch (self.kind) {
            .slice_bf16 => try self.rows.sliceBf16(self.rig.s, self.dev_a, 64, self.dev_out, 40, self.shape.rows),
            .interleave2_bf16 => try self.rows.interleave2Bf16(self.rig.s, self.dev_a, self.dev_out, 32, self.shape.rows),
            .slice_f32 => try self.rows.sliceF32(self.rig.s, self.dev_a, 48, 12, self.dev_out, 24, self.shape.rows),
            .splice_streams => {
                const out_view: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.out_bytes };
                try out_view.fill8(0x5a, self.rig.s.handle);
                try self.rows.spliceStreams(self.rig.s, self.dev_a, self.dev_b, self.dev_out, 64, self.shape.rows);
            },
            .vit_merge_gather => try self.rows.vitMergeGather(self.rig.s, self.dev_a, self.dev_out, self.shape.rows, self.shape.n, self.shape.k),
            .vit_append_newline => try self.rows.vitAppendNewline(self.rig.s, self.dev_a, self.dev_b, self.dev_out, self.shape.rows, self.shape.n, self.shape.k),
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const CopyCase) usize {
        return self.out_bytes;
    }
    pub fn reference(_: *CopyCase, _: []f64, _: []f64) !void {}
    pub fn replaced(self: *CopyCase) ![]const u8 {
        return self.want.items;
    }
    pub fn check(self: *CopyCase, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        if (!std.mem.eql(u8, got, self.want.items)) return "hy4_rows byte-copy output differs from exact reference";
        if (!try self.rig.intact()) return "hy4_rows byte-copy overwrote guard region";
        return null;
    }
};
