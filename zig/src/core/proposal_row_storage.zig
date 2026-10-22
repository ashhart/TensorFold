//! Exact stored row planes and owned layout, independent of backend selection module ownership.
const std = @import("std");
const pv = @import("proposal_vocab.zig");
pub const Storage = enum { bf16, affine, unsupported };
pub const Recipe = enum { qlinear_decode, qlinear_prompt, affine_mm, unsupported };
pub const SideDtype = enum { unknown, bf16, f16, f32 };
pub const Plane = struct { bytes: []const u8 = &.{}, stride: usize = 0 };
pub const Shape = struct { dtype: enum { bf16, f32 }, vocab: u32, logical: u32, stride: u32, ids: ?[]const u32 };

/// Storage accepts its metadata view type, so pack-only users do not import a backend's named selectors.
pub fn Views(comptime ScoreView: type) type {
    return struct {
        pub const Source = struct {
            identity: pv.Identity,
            storage: Storage,
            recipe: Recipe,
            codes: Plane,
            scales: Plane = .{},
            biases: Plane = .{},
            scale_dtype: SideDtype = .unknown,
            bias_dtype: SideDtype = .unknown,
            ranks: u32 = 1,
            queue: u64 = 0,
            tied: bool = false,
            output_f32: bool = false,

            fn layout(s: Source, a: *const pv.Artifact, tile: u32) !Layout {
                if (!a.matches(s.identity)) return error.ProposalIdentityMismatch;
                if (s.ranks != 1) return error.ProposalPlacementUnsupported;
                if (s.storage == .unsupported or s.recipe == .unsupported) return error.ProposalBasisUnsupported;
                const h = s.identity.hidden;
                var bits: u8 = 16;
                var side: usize = 0;
                switch (s.storage) {
                    .bf16 => if (!std.mem.eql(u8, s.identity.format, "bf16") or s.identity.group != 0) return error.ProposalBasisUnsupported,
                    .affine => {
                        if (s.scale_dtype != .bf16 or s.bias_dtype != .bf16) return error.ProposalBasisUnsupported;
                        if (!std.mem.startsWith(u8, s.identity.format, "affine")) return error.ProposalBasisUnsupported;
                        bits = std.fmt.parseInt(u8, s.identity.format[6..], 10) catch return error.ProposalBasisUnsupported;
                        if ((std.mem.indexOfScalar(u8, &.{ 2, 3, 4, 5, 6, 8 }, bits) == null and !(a.isLegacy() and bits == 7)) or
                            s.identity.group == 0 or h % s.identity.group != 0) return error.ProposalBasisUnsupported;
                        side = try std.math.mul(usize, h / s.identity.group, 2);
                    },
                    .unsupported => unreachable,
                }
                const row_bits = try std.math.mul(usize, h, bits);
                if (row_bits == 0 or row_bits % 8 != 0) return error.ProposalRowShape;
                const row = row_bits / 8;
                try plane(s.codes, s.identity.vocab, row);
                if (side == 0) {
                    if (s.scales.bytes.len != 0 or s.biases.bytes.len != 0) return error.ProposalRowShape;
                } else {
                    try plane(s.scales, s.identity.vocab, side);
                    try plane(s.biases, s.identity.vocab, side);
                }
                const physical = try a.physicalCount(tile);
                const per_row = try std.math.add(usize, row, try std.math.mul(usize, side, 2));
                const bytes = try std.math.add(usize, try std.math.mul(usize, physical, per_row), try std.math.mul(usize, a.ids().len, 4));
                return .{ .row = row, .side = side, .physical = physical, .bits = bits, .bytes = bytes };
            }
            pub fn price(s: Source, a: *const pv.Artifact, tile: u32) !usize {
                return (try s.layout(a, tile)).bytes;
            }
            pub fn extract(s: Source, gpa: std.mem.Allocator, a: *const pv.Artifact, tile: u32, budget: usize) !Owned {
                const l = try s.layout(a, tile);
                if (l.bytes > budget) return error.ProposalRowBudget;
                const ids = try gpa.dupe(u32, a.ids());
                errdefer gpa.free(ids);
                const codes = try copy(gpa, s.codes, ids, l.row, l.physical);
                errdefer gpa.free(codes);
                const scales = try copy(gpa, s.scales, ids, l.side, l.physical);
                errdefer gpa.free(scales);
                const biases = try copy(gpa, s.biases, ids, l.side, l.physical);
                return .{ .gpa = gpa, .ids = ids, .codes = codes, .scales = scales, .biases = biases, .vocab = s.identity.vocab, .hidden = s.identity.hidden, .bits = l.bits, .group = s.identity.group, .storage = s.storage, .recipe = s.recipe, .physical = l.physical, .fingerprint = a.fingerprint, .queue = s.queue, .tied = s.tied, .output_f32 = s.output_f32 };
            }
        };
        const Layout = struct { row: usize, side: usize, physical: u32, bits: u8, bytes: usize };
        fn plane(p: Plane, rows: u32, row: usize) !void {
            if (rows == 0 or p.stride < row) return error.ProposalRowShape;
            const end = try std.math.add(usize, try std.math.mul(usize, rows - 1, p.stride), row);
            if (end > p.bytes.len) return error.ProposalRowShape;
        }
        fn copy(gpa: std.mem.Allocator, p: Plane, ids: []const u32, row: usize, physical: u32) ![]align(16) u8 {
            const out = try gpa.alignedAlloc(u8, .@"16", try std.math.mul(usize, physical, row));
            @memset(out, 0);
            if (row > 0) for (ids, 0..) |id, i| @memcpy(out[i * row ..][0..row], p.bytes[id * p.stride ..][0..row]);
            return out;
        }
        pub const Owned = struct {
            gpa: std.mem.Allocator,
            ids: []const u32,
            codes: []align(16) u8,
            scales: []align(16) u8,
            biases: []align(16) u8,
            vocab: u32,
            hidden: u32,
            bits: u8,
            group: u32,
            storage: Storage,
            recipe: Recipe,
            physical: u32,
            fingerprint: [32]u8,
            queue: u64,
            tied: bool,
            output_f32: bool,
            pub fn deinit(v: *Owned) void {
                v.gpa.free(v.ids);
                v.gpa.free(v.codes);
                v.gpa.free(v.scales);
                v.gpa.free(v.biases);
                v.* = undefined;
            }
            pub fn scores(v: *const Owned) ScoreView {
                return .{ .vocab = v.vocab, .logical = @intCast(v.ids.len), .stride = v.physical, .ids = v.ids, .dtype = if (v.output_f32) .f32 else .bf16 };
            }
            /// Existing affine_mm pipelines only, under the same stored format and numerical recipe as the full head.
            pub fn affine(v: *const Owned, comptime Matmul: type, format: anytype, e: anytype, pipes: anytype, buffers: anytype, x: anytype, sums: anytype, y: anytype, rows: u32) !void {
                if (v.storage != .affine or v.recipe != .affine_mm or format.bits != v.bits or format.group != v.group or format.out_f32 != v.output_f32 or
                    (v.group != 32 and v.group != 64) or v.physical % 64 != 0 or rows == 0) return error.ProposalBasisUnsupported;
                const q = .{ .w = buffers.w, .s = buffers.s, .b = buffers.b, .n = v.physical, .k = v.hidden };
                Matmul.dense(e, pipes, x, sums, q, y, rows);
            }
        };
    };
}
