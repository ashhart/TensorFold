//! Real HC launch selection at the driver boundary; no device and no numerical emulation.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const t = std.testing;

const Record = struct { id: usize, gx: u32, gy: u32, bx: u32, rows: i64 = 0 };
var calls: [2]Record = undefined;
var count: usize = 0;
fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
    const id = @intFromPtr(f);
    calls[count] = .{ .id = id, .gx = gx, .gy = gy, .bx = bx };
    if (id == 2) calls[count].rows = @as(*const i64, @ptrCast(@alignCast(args.?[5].?))).*;
    count += 1;
    return 0;
}

fn actual(on: bool, prompt: bool, rows: usize) !void {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = launch;
    driver.api.cuLaunchKernelEx = glm.pdl.forwardEx(launch);
    var k: glm.glue.Kernels = undefined;
    inline for (@typeInfo(glm.glue.Kernels).@"struct".field_names) |name| @field(k, name) = .{ .d = &driver, .handle = null };
    k.hc_partial8.handle = @ptrFromInt(1);
    k.hc_partial8_rows.handle = @ptrFromInt(2);
    k.hc_finish.handle = @ptrFromInt(3);
    k.hc_partial.handle = @ptrFromInt(4);
    count = 0;
    try glm.glue.hcPre(.{ .glue = &k, .hc_rows = on, .prompt = prompt }, .{ .d = &driver, .handle = null }, 256, 512, 768, 1024, .{ .bf16 = 1280 }, 1536, 1792, 2048, 2304, 2560, 1e-6, 1e-6, 20, rows, 4096, 4);
}

test "GLM HC rows: actual opt-in launch shares all decode rows and keeps the finish unchanged" {
    for (1..33) |rows| {
        try actual(true, false, rows);
        try t.expectEqual(@as(usize, 2), count);
        try t.expectEqualDeep(Record{ .id = 2, .gx = 1, .gy = 16, .bx = 1024, .rows = @intCast(rows) }, calls[0]);
        try t.expectEqualDeep(Record{ .id = 3, .gx = @intCast(rows), .gy = 1, .bx = 256 }, calls[1]);
    }
}

test "GLM HC rows: default, explicit off, prompts and wide calls retain original launches" {
    try t.expect(!(@as(glm.glue.Set, .{ .glue = undefined })).hc_rows);
    for ([_]usize{ 1, 2, 8, 16, 17, 31, 32, 33, 63, 64, 65, 2048 }) |rows| {
        for ([_][2]bool{ .{ false, false }, .{ false, true }, .{ true, true } }) |mode| {
            try actual(mode[0], mode[1], rows);
            try t.expectEqual(@as(usize, 2), count);
            try t.expectEqual(@as(usize, if (rows < 64) 1 else 4), calls[0].id);
            try t.expectEqual(@as(u32, @intCast(if (rows < 64) rows else (rows + 3) / 4)), calls[0].gx);
        }
        if (rows > 32) {
            try actual(true, false, rows);
            try t.expectEqual(@as(usize, if (rows < 64) 1 else 4), calls[0].id);
        }
    }
}

test "GLM HC rows: contract registers every requested width and a real wrong-row control" {
    const fixture = @import("tests/cuda/contract_glm_hc_rows.zig");
    var seen: [33]bool = @splat(false);
    for (fixture.served) |s| {
        try t.expect(s.n == 25 and s.k == 16384 and s.rows <= 32);
        seen[s.rows] = true;
    }
    for (1..33) |r| try t.expect(seen[r]);
    try t.expect(seen[17] and seen[31] and seen[32]);
    var f: fixture.Fixture = undefined;
    f.fault = false;
    try t.expect(f.kernel().spec.replaces != null and f.kernel().control == null);
    f.fault = true;
    try t.expect(f.kernel().control == .shapes);
    const registry = @import("core").contract.registry;
    var records: usize = 0;
    for (registry.cuda) |r| try t.expect(!std.mem.startsWith(u8, r.launch.symbol, "tf_glm_hc_partial8_rows"));
    for (registry.cuda_pending) |r| if (std.mem.startsWith(u8, r.launch.symbol, "tf_glm_hc_partial8_rows")) {
        records += 1;
        try t.expectEqual(@as(usize, @import("core").contract.limits.sms.len), try registry.checkCuda(t.allocator, &.{r}, true));
        try t.expect(r.launch.threads == 1024 and r.launch.dynamic == 0);
        try t.expectEqualDeep(registry.none, registry.usageFor(r, .{ .major = 13, .minor = 3 }));
    };
    try t.expectEqual(@as(usize, 2), records);
}
