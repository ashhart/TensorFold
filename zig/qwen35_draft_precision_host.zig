//! Production drafting projections recorded by an inert CUDA driver; arithmetic witnesses run on the CPU.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const fixture = @import("qwen35_verify_host.zig");
const P = q.draft.precision;
const Trace = fixture.Trace;
const T = std.testing;
const x = 0x10000;
const xs = 0x20000;
const out = 0x30000;
const k = 8192;
const n = 4096;
const weight: cuda.qlinear.Affine4 = .{ .w = 10, .s = 20, .b = 30, .n = n, .k = k, .npad = n };

fn reset() void {
    Trace.count = 0;
    Trace.prompts = 0;
}
/// Native rows in the fewest verify tiles (decode's bits a row), never the prompt GEMM unless the role is a prompt's.
fn check(rows: usize, role: P.Role, tile: usize) !void {
    const prompt = role == .prompt and rows > 16;
    try T.expectEqual(@as(usize, @intFromBool(prompt)), Trace.prompts);
    try T.expectEqual(if (prompt) @as(usize, 0) else (rows + tile - 1) / tile, Trace.count);
    for (Trace.calls[0..Trace.count], 0..) |call, i| {
        const at = i * tile;
        const m = @min(tile, rows - at);
        try T.expectEqual(@as(c_int, @intCast(m)), call.rows);
        try T.expectEqual(@as(u64, x + at * k * 2), call.x);
        try T.expectEqual(@as(u64, xs + at * (k / 64) * 4), call.xs);
        try T.expectEqual(@as(u64, out + at * n * 2), call.out);
        try T.expect(call.id != 4);
        if (m > 16) try T.expectEqual(@as(usize, if (m <= 32) 5 else if (m <= 64) 6 else 7), call.id);
    }
}

test "Draft precision: DFlash projections preserve each role across row and stream boundaries" {
    for ([_]bool{ false, true }) |gb10| {
        var driver: cuda.Driver = undefined;
        var e = fixture.engine(&driver, gb10);
        var d: q.draft.Draft = undefined;
        d.e = &e;
        d.xs = xs;
        for ([_]P.Role{ .prompt, .live, .proposal }) |role| {
            d.projection_role = role;
            for ([_]usize{ 16, 17, 32, 64, 65 }) |rows| for ([_]usize{ 1, 2, 4 }) |streams| {
                reset();
                try d.proj(weight, x, out, rows * streams);
                try check(rows * streams, role, e.affine.verifyRows());
                reset();
                try d.projAgain(weight, x, out, rows * streams);
                try check(rows * streams, role, e.affine.verifyRows());
            };
        }
    }
}

test "Draft precision: MTP projections and the DFlash vocabulary head retain native proposal tiles" {
    for ([_]bool{ false, true }) |gb10| {
        var driver: cuda.Driver = undefined;
        var e = fixture.engine(&driver, gb10);
        var d: q.mtp.Mtp = undefined;
        d.e = &e;
        d.xs = xs;
        const l: q.engine.Proj = .{ .a4 = weight };
        for ([_]P.Role{ .prompt, .live, .proposal }) |role| {
            d.projection_role = role;
            for ([_]usize{ 16, 17, 32, 64, 65 }) |rows| for ([_]usize{ 1, 2, 4 }) |streams| {
                reset();
                try d.proj(l, x, out, rows * streams);
                try check(rows * streams, role, e.affine.verifyRows());
                reset();
                try P.project(&e, l, x, xs, out, rows * streams, .proposal, P.dflash_projection_prompt_cutoff);
                try check(rows * streams, .proposal, e.affine.verifyRows());
            };
        }
    }
}

const Moe = struct {
    g: struct { hidden: usize = 256 } = .{},
    count: usize = 0,
    prompt: bool = false,
    total: usize = 0,
    pub fn routedRows(e: *Moe, router: u64, experts: u32, xa: u64, oa: u64, rows: usize, wide: bool, _: u64) !void {
        try T.expectEqual(@as(u64, 7), router);
        try T.expectEqual(@as(u32, 9), experts);
        try T.expectEqual(x + e.total * 256 * 2, xa);
        try T.expectEqual(out + e.total * 256 * 2, oa);
        if (!wide) try T.expect(rows <= 16);
        e.prompt = wide;
        e.count += 1;
        e.total += rows;
    }
};
test "Draft precision: live MTP absorbs and wide internal levels retain fp32 expert-down selection" {
    for ([_]P.Role{ .prompt, .live, .proposal }) |role| {
        for ([_]usize{ 16, 17, 32, 64, 65 }) |rows| for ([_]usize{ 1, 2, 4 }) |streams| {
            var e: Moe = .{};
            try P.routed(&e, 7, @as(u32, 9), x, out, rows * streams, role);
            const prompt = role == .prompt and rows * streams > 16;
            try T.expectEqual(prompt, e.prompt);
            try T.expectEqual(if (prompt) @as(usize, 1) else (rows * streams + 15) / 16, e.count);
            try T.expectEqual(rows * streams, e.total);
        };
    }
}

fn bf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}
test "Draft precision: valid affine code witness distinguishes the forbidden extra weight round" {
    const scale: f32 = 0.71484375;
    const bias: f32 = -2.46875;
    const input: f32 = -1.2265625;
    const native = bf16(input * scale + input * bias);
    const prompt = bf16(input * bf16(scale + bias));
    try T.expectEqual(@as(f32, 2.15625), native);
    try T.expectEqual(@as(f32, 2.140625), prompt);
    try T.expect(native != prompt);
}

test {
    _ = @import("qwen35_draft_absorb_host.zig");
}

test "Draft precision: the actual grouped MTP QKV keeps live and proposal roles at 17 32 and 128 rows" {
    for (0..3) |mode| {
        const gb10 = mode == 2;
        var driver: cuda.Driver = undefined;
        var e = fixture.engine(&driver, gb10);
        if (mode == 1) {
            e.affine.seg_fn = .{ .d = &driver, .handle = @ptrFromInt(8) };
            e.affine.split = .{ .work = .{ .d = &driver, .ptr = 0x9000000, .len = 1 << 25 }, .tickets = .{ .d = &driver, .ptr = 0xa000000, .len = 4096 } };
        }
        var d: q.mtp.Mtp = undefined;
        d.e = &e;
        d.x = x;
        d.xs = xs;
        d.qraw = out;
        d.kraw = out + 0x1000000;
        d.vb = out + 0x2000000;
        d.w.q = .{ .a4 = weight };
        d.w.k = .{ .a4 = weight };
        d.w.v = .{ .a4 = weight };
        for ([_]P.Role{ .live, .proposal, .prompt }) |role| {
            d.projection_role = role;
            for ([_]usize{ 17, 32, 128 }) |m| {
                reset();
                try d.projQkv(m);
                const prompt = role == .prompt;
                const per = (m + e.affine.verifyRows() - 1) / e.affine.verifyRows();
                try T.expectEqual(@as(usize, if (prompt) 3 else 0), Trace.prompts);
                try T.expectEqual(if (prompt) @as(usize, 0) else 3 * per, Trace.count);
                for (Trace.calls[0..Trace.count], 0..) |call, i| {
                    const at = (if (mode == 1) i / 3 else i % per) * e.affine.verifyRows();
                    const projection = if (mode == 1) i % 3 else i / per;
                    try T.expectEqual(@as(u64, x + at * k * 2), call.x);
                    try T.expectEqual(@as(u64, xs + at * (k / 64) * 4), call.xs);
                    try T.expectEqual(@as(u64, out + projection * 0x1000000 + at * n * 2), call.out);
                }
            }
        }
    }
}
