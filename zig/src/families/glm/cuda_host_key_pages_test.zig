//! Record the actual MTP prompt call's separate host-latent and device-key page bindings without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const dsa = @import("cuda_dsa_block.zig");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");
const t = std.testing;
const host_page = 6_684_672;
const key_page = 393_216;

const Capture = struct {
    const Seen = struct { pool: u64 = 0, score: u64 = 0, latent: u64 = 0, token_rows: u64 = 0 };
    var seen: Seen = .{};
    fn arg(args: ?[*]?*anyopaque, i: usize) u64 {
        const p: [*]const u8 = @ptrCast(args.?[i].?);
        return std.mem.readInt(u64, p[0..8], .little);
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        switch (@intFromPtr(f)) {
            2 => seen.pool = arg(args, 7),
            3 => seen.score = arg(args, 13),
            5 => seen.token_rows = arg(args, 6),
            7 => seen.latent = arg(args, 6),
            else => {},
        }
        return cuda.abi.success;
    }
    fn symbol(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, 5) else 100);
        return cuda.abi.success;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

fn run(rows: usize, pks: u64) !Capture.Seen {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    d.api.cuModuleGetFunction = Capture.symbol;
    const f = function(&d, 100);
    var raw: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| @field(raw, name) = f;
    raw.pool_keys = function(&d, 2);
    raw.scores = function(&d, 3);
    raw.lat_write = function(&d, 7);
    const kernels: mm.Kernels = .{ .by_rows = @splat(f), .packed_rows = @splat(@splat(f)), .prompt = f, .reduce = f, .tile = f, .tile_packed = @splat(f), .pdl = false };
    const l: dsa.Layer = .{ .proj = .{ .bf16 = 0x100000000 }, .q_norm = 0, .kv_norm = 0, .q_b = .{ .bf16 = 0x200000000 }, .wk = 0, .wv = 0, .o = .{ .bf16 = 0x300000000 }, .index = .{ .kw = .{ .bf16 = 0x400000000 }, .gate = 0, .ln_w = 0, .ln_b = 0, .ape = 0, .qb = .{ .bf16 = 0x500000000 }, .heads = 32, .d = 128 }, .heads = 32, .qk = 256, .dv = 256, .lw = 512, .q_lora = 1536, .dims = 4096, .eps = 1e-6, .dense_limit = 2048 };
    const c: dsa.Caches = .{ .lc = 0x600000000, .ik = 0, .ig = 0, .pk = 0x700000000, .pk_rows = 16386, .pt = 0x800000000, .ps = host_page, .pks = pks };
    const s: dsa.Scratch = .{ .dp = 0, .qr = 0, .xs_qr = 0, .lat = 0, .xs_lat = 0, .q = 0, .ikr = 0, .igr = 0, .router_part = 0, .qa = 0, .ol = 0, .vn = 0, .qi = 0, .partials = .{ .po = 0, .pm = 0, .pl = 0 }, .dummy = 0, .select = .{ .scores = 0, .ls = 0, .li = 0, .pools = 0, .at = 0 }, .tokens = 0, .counts = 0, .mm_part = 0, .at = 0 };
    Capture.seen = .{};
    try dsa.blockLastRow(.{ .set = .{ .glue = &raw, .mm = &kernels, .prompt = true }, .ops = .{ .d = &d, .handle = null } }, .{ .d = &d, .handle = null }, l, c, s, 0, 4096, rows, 0, 0, (rows + 511) / 512, 0);
    return Capture.seen;
}

test "GLM host key pages: MTP writes use the device pooled-key stride, not the host latent stride" {
    for ([_]u64{ 0, key_page }) |pks| for ([_]usize{ 1025, 2047, 2048, 2049, 8193 }) |rows| {
        const got = try run(rows, pks);
        try t.expectEqual(@as(u64, if (pks == 0) host_page else key_page), got.pool);
        try t.expectEqual(@as(u64, host_page), got.latent);
    };
}

test "GLM host key pages: last-row scores use key pages while token IDs retain host physical row units" {
    for ([_]u64{ 0, key_page }) |pks| for ([_]usize{ 2049, 8193 }) |rows| {
        const got = try run(rows, pks);
        try t.expectEqual(@as(u64, if (pks == 0) host_page else key_page), got.score);
        try t.expectEqual(@as(u64, host_page / 1024), got.token_rows);
    };
    const arena = 145 * @as(u64, key_page);
    const mtp_offset = 11 * 128 * 128 * 2;
    try t.expect(mtp_offset + 4 * @as(u64, host_page) + 32768 < arena);
    try t.expect(mtp_offset + 16 * @as(u64, host_page) > arena);
    try t.expect(mtp_offset + 16 * @as(u64, key_page) + 32768 < arena);
}
