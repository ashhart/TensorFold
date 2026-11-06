//! Recording drivers execute the production CUDA/Metal learner launch ABI over owned host buffers.
const std = @import("std");
const cu = @import("cuda");
const core = @import("core");
const F = @import("qwen4").family;
const fixture = F.learning_fixture;
const a = std.testing.allocator;
const Params = core.living_rows.Params;
extern fn tf_living_rows_cpu(c_uint, Params, [*][*]u8) void;
extern fn tf_living_gdn_cpu(c_uint, Params, [*][*]u8) void;
extern fn tf_living_gemma_cpu(c_uint, Params, [*][*]u8) void;
extern fn tf_living_qsa_cpu(c_uint, Params, [*][*]u8) void;
fn id(name: []const u8) ?usize {
    inline for (.{ core.living_rows, core.living_gdn, core.living_qsa, core.living_gemma }, 0..) |spec, kind| {
        for (spec.names, 0..) |symbol, op| if (std.mem.eql(u8, name, symbol)) return kind * 100 + op;
    }
    return null;
}
fn run(symbol: usize, p: Params, ptrs: *[11][*]u8) void {
    switch (symbol / 100) {
        0 => tf_living_rows_cpu(@intCast(symbol % 100), p, ptrs),
        1 => tf_living_gdn_cpu(@intCast(symbol % 100), p, ptrs),
        2 => tf_living_qsa_cpu(@intCast(symbol % 100), p, ptrs),
        3 => tf_living_gemma_cpu(@intCast(symbol % 100), p, ptrs),
        else => unreachable,
    }
}
const M = struct {
    pub const Size = struct {
        width: usize,
        height: usize,
        depth: usize,
        pub fn of(w: usize, h: usize, d: usize) @This() {
            return .{ .width = w, .height = h, .depth = d };
        }
    };
    pub const Device = struct { id: usize = 1 };
    pub const Buffer = struct {
        data: []u8,
        owner: usize = 1,
        pub fn deviceId(b: @This()) usize {
            return b.owner;
        }
        pub fn gpuAddress(b: @This()) u64 {
            return @intFromPtr(b.data.ptr);
        }
        pub fn length(b: @This()) usize {
            return b.data.len;
        }
    };
    pub const CompileOptions = struct {
        pub fn mlx() @This() {
            return .{};
        }
    };
    pub const Library = struct {
        pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !@This() {
            if (std.mem.indexOf(u8, text, "TF_LIVING_ROWS") == null or std.mem.indexOf(u8, text, "lw_rows") == null) return error.Source;
            return .{};
        }
        pub fn deinit(_: @This()) void {}
    };
    var live: usize = 0;
    var calls: usize = 0;
    var fail_at: ?usize = null;
    var selected: usize = 0;
    var p: Params = undefined;
    var ptrs: [11][*]u8 = undefined;
    pub const Pipeline = struct {
        symbol: usize,
        pub fn init(_: Device, _: Library, name: []const u8, _: bool) !@This() {
            const symbol = id(name) orelse return error.Symbol;
            if (fail_at == symbol) return error.PlantedPipeline;
            live += 1;
            return .{ .symbol = symbol };
        }
        pub fn maxThreads(_: @This()) usize {
            return 128;
        }
        pub fn deinit(_: @This()) void {
            live -= 1;
        }
    };
    pub const ComputeEncoder = struct {
        owner: usize = 1,
        pub fn deviceId(e: @This()) usize {
            return e.owner;
        }
        pub fn setPipeline(_: @This(), pipe: Pipeline) void {
            selected = pipe.symbol;
        }
        pub fn setBuffer(_: @This(), b: Buffer, offset: usize, index: usize) void {
            ptrs[index] = b.data.ptr + offset;
        }
        pub fn setBytes(_: @This(), bytes: []const u8, index: usize) void {
            std.debug.assert(index == 11 and bytes.len == 80);
            p = std.mem.bytesToValue(Params, bytes);
        }
        pub fn dispatchThreads(_: @This(), grid: Size, threads: Size) void {
            std.debug.assert(grid.width == p.u[15] and grid.height == 1 and grid.depth == 1 and threads.width == 128 and threads.height == 1 and threads.depth == 1);
            calls += 1;
            run(selected, p, &ptrs);
        }
        pub fn barrier(_: @This()) void {}
    };
};
const Trace = struct {
    var live: usize = 0;
    var calls: usize = 0;
    var fail_at: ?usize = null;
    fn load(out: *cu.abi.Module, _: ?*const anyopaque) callconv(.c) cu.abi.Result {
        live += 1;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn unload(_: cu.abi.Module) callconv(.c) cu.abi.Result {
        live -= 1;
        return 0;
    }
    fn function(out: *cu.abi.Function, _: cu.abi.Module, name: [*:0]const u8) callconv(.c) cu.abi.Result {
        const symbol = id(std.mem.span(name)) orelse return cu.abi.error_not_found;
        out.* = @ptrFromInt(symbol + 1);
        return 0;
    }
    fn attribute(out: *c_int, _: cu.abi.FunctionAttribute, fun: cu.abi.Function) callconv(.c) cu.abi.Result {
        out.* = if (fail_at == @intFromPtr(fun) - 1) 64 else 128;
        return 0;
    }
    fn read(comptime T: type, args: ?[*]?*anyopaque, index: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[index].?))).*;
    }
    fn launch(fun: cu.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cu.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cu.abi.Result {
        const p = read(Params, args, 11);
        std.debug.assert(gx == (p.u[15] + 127) / 128 and gy == 1 and gz == 1 and bx == 128 and by == 1 and bz == 1 and shared == 0);
        var ptrs: [11][*]u8 = undefined;
        for (&ptrs, 0..) |*ptr, i| ptr.* = @ptrFromInt(read(u64, args, i));
        calls += 1;
        run(@intFromPtr(fun) - 1, p, &ptrs);
        return 0;
    }
};
fn driver() cu.Driver {
    var d: cu.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleLoadData = Trace.load;
    d.api.cuModuleUnload = Trace.unload;
    d.api.cuModuleGetFunction = Trace.function;
    d.api.cuFuncGetAttribute = Trace.attribute;
    d.api.cuLaunchKernel = Trace.launch;
    return d;
}
fn params(comptime spec: type, op: spec.Op) Params {
    if (spec == core.living_gemma) return F.learning_contract_data.params(spec, op, 2, 64);
    var p: Params = .{};
    if (spec == core.living_rows) {
        p.u[0] = 2;
        p.u[1] = 64;
        p.u[2] = 2;
        p.u[4] = 64;
        p.f[0] = 1e-6;
        switch (op) {
            .gate => {
                p.u[2] = 32;
                p.u[3] = 128;
            },
            .rope_rms => {
                p.u[1] = 2;
                p.u[2] = 32;
                p.u[3] = 16;
                p.u[4] = 128;
                p.u[6] = 64;
            },
            .pocket_act, .pocket_add, .pocket_back, .pocket_coeff => {
                p.f[0] = 1;
            },
            else => {},
        }
    } else if (spec == core.living_gdn) {
        p.u[0] = 2;
        p.u[1] = 1;
        p.u[2] = 1;
        p.u[3] = 514;
        p.f[0] = 1e-6;
    } else {
        p.u[0] = 2;
        p.u[1] = 2;
        p.u[2] = 1;
        p.u[3] = 4;
        p.u[4] = 2;
        p.f[0] = 0.5;
    }
    return p;
}
fn check(comptime spec: type, kind: usize) !void {
    const C = F.train_rows_cuda.Adapter(cu, spec);
    const Metal = F.train_rows_metal.Adapter(M, spec);
    var d = driver();
    const image: [8]u8 align(8) = @splat(1);
    const stream = cu.Stream.borrowed(&d, null);
    var ck = try C.load(&d, &image);
    defer ck.deinitDrained();
    var mk = try Metal.load(a, .{});
    defer mk.deinitDrained();
    for (std.enums.values(spec.Op)) |op| {
        var c = try fixture.Case(spec).init(op, params(spec, op));
        defer c.deinit();
        var expected = try fixture.Case(spec).init(op, params(spec, op));
        defer expected.deinit();
        try expected.run();
        var cv: [11]C.View = undefined;
        var mv: [11]Metal.View = undefined;
        for (c.data, &cv, &mv) |data, *cc, *mm| {
            cc.* = .{ .buffer = .{ .d = &d, .ptr = @intFromPtr(data.ptr), .len = data.len }, .len = data.len };
            mm.* = .{ .buffer = .{ .data = data }, .len = data.len };
        }
        try ck.encode(stream, op, c.p, cv);
        for (8..11) |i| try std.testing.expectEqualSlices(u8, expected.data[i], c.data[i]);
        for (c.data) |data| @memset(data, 0);
        try mk.encode(.{}, op, c.p, mv);
        for (8..11) |i| try std.testing.expectEqualSlices(u8, expected.data[i], c.data[i]);
        const calls = M.calls + Trace.calls;
        var bad = mv;
        bad[0].buffer.owner = 2;
        try std.testing.expectError(error.LivingRowsContext, mk.encode(.{}, op, c.p, bad));
        var foreign = d;
        var cbad = cv;
        cbad[0].buffer.d = &foreign;
        try std.testing.expectError(error.LivingRowsContext, ck.encode(stream, op, c.p, cbad));
        try std.testing.expectEqual(calls, M.calls + Trace.calls);
    }
    const kept_c = Trace.live;
    const kept_m = M.live;
    Trace.fail_at = kind * 100 + 1;
    try std.testing.expectError(error.LivingRowsThreads, C.load(&d, &image));
    Trace.fail_at = null;
    M.fail_at = kind * 100 + 1;
    try std.testing.expectError(error.PlantedPipeline, Metal.load(a, .{}));
    M.fail_at = null;
    try std.testing.expectEqual(kept_c, Trace.live);
    try std.testing.expectEqual(kept_m, M.live);
}
test "Living native launch: every row GDN and QSA ABI, context refusal and partial owner cleanup" {
    inline for (.{ core.living_rows, core.living_gdn, core.living_qsa, core.living_gemma }, 0..) |spec, kind| try check(spec, kind);
    try std.testing.expectEqual(@as(usize, 0), M.live);
    try std.testing.expectEqual(@as(usize, 0), Trace.live);
}
