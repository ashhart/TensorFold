//! Dense Qwen4 uses native embedded images and one stable CUDA context through its shared lane host.
const std = @import("std");
const C = @import("cuda");
const Api = @import("engine_api");
const L = @import("lanes");
const F = @import("qwen4").family;
const G = F.cuda_gpu.Gpu(C);
const B = G.Storage;
const P = struct {
    const Self = @This();
    pub const Engine = F.engine.Engine(G);
    pub const Drawer = F.cuda_draw.Drawer(C, L, B);
    a: std.mem.Allocator,
    driver: C.Driver,
    context: C.Context,
    storage: B,
    pub fn open(a: std.mem.Allocator, _: std.Io, o: Api.Open) !*Self {
        const p = try a.create(Self);
        errdefer a.destroy(p);
        p.a = a;
        p.driver = try C.Driver.open();
        errdefer p.driver.close();
        const text = @import("cuda_device.zig").getenv("TF_CUDA_DEVICE") orelse "0";
        const trimmed = std.mem.trim(u8, text, " ");
        const device = o.device orelse (if (trimmed.len == 0) 0 else std.fmt.parseInt(u32, trimmed, 10) catch return error.Qwen4Device);
        p.context = try C.Context.init(&p.driver, std.math.cast(c_int, device) orelse return error.Qwen4Device);
        errdefer p.context.deinit();
        const stream = try C.Stream.init(&p.driver, true);
        p.storage = .{ .driver = &p.driver, .context = &p.context, .stream = stream };
        return p;
    }
    pub fn room(p: *Self, a: std.mem.Allocator, io: std.Io) !u64 {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        var why: []const u8 = "";
        const pool = try @import("cuda_family_memory.zig").pool(arena.allocator(), io, &p.context, &why) orelse return error.Qwen4MemoryBudget;
        return pool.room(C.usage(false).device);
    }
    const Loader = struct {
        pub fn load(_: @This(), a: std.mem.Allocator, b: *B, c: *const F.Config, problem: *F.Problem) !G {
            return G.load(a, b, c, problem, .{ .projection = C.kernels.qwen4_projection, .hc_scalar = C.kernels.qwen4_hc, .hc = C.kernels.flashnext_hc, .gdn = C.kernels.qwen4_gdn, .attention = C.kernels.flashnext_attn, .batch = C.kernels.flashnext_batch, .ple = C.kernels.qwen4_ple }, .hf);
        }
    };
    pub fn loader(_: *Self) Loader {
        return .{};
    }
    pub fn drawer(p: *Self, _: std.mem.Allocator, vocab: usize, rows: usize, streams: usize) !Drawer {
        return Drawer.load(&p.storage, vocab, rows, streams, C.kernels.sample, C.kernels.torch_argmax);
    }
    pub fn healthy(p: *const Self) bool {
        return p.context.healthy();
    }
    pub fn close(p: *Self) !void {
        try p.storage.drain();
        p.storage.stream.deinit();
        p.context.deinit();
        p.driver.close();
        const a = p.a;
        a.destroy(p);
    }
};
pub const Native = F.native_host.Host(P, Api, L);
pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: Api.Open, problem: *[]const u8) !?Api.Opened {
    if (F.native_host.refusal(o)) |why| {
        problem.* = why;
        return null;
    }
    if (!C.kernels.available) {
        problem.* = "Qwen4 CUDA requires the embedded native kernel images";
        return null;
    }
    if (@import("cuda_device.zig").getenv("TENSORFOLD_CUDA_KERNELS")) |v| if (v.len > 0 and !std.mem.eql(u8, v, "native")) {
        problem.* = "Qwen4 uses embedded native images: unset TENSORFOLD_CUDA_KERNELS or set native";
        return null;
    };
    var p: F.Problem = .{};
    return Native.open(gpa, io, o, &p) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "Qwen4 CUDA refused {s}: {s}", .{ if (p.name().len > 0) p.name() else "native load", @errorName(err) });
        return null;
    };
}
