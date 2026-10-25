//! Dense Qwen4 owns one Metal queue and retained compute/blit session for the shared native lane host.
const std = @import("std");
const M = @import("metal");
const Api = @import("engine_api");
const L = @import("tensorfold").lanes;
const F = @import("qwen4").family;
const G = F.metal_gpu.Gpu(M);
const B = G.Storage;
const P = struct {
    const Self = @This();
    pub const Engine = F.engine.Engine(G);
    pub const Drawer = F.metal_draw.Drawer(M, L, B);
    a: std.mem.Allocator,
    commands: M.CommandSession,
    storage: B,
    pub fn open(a: std.mem.Allocator, _: std.Io, o: Api.Open) !*Self {
        if (o.device != null and o.device.? != 0) return error.Qwen4Device;
        const p = try a.create(Self);
        errdefer a.destroy(p);
        p.a = a;
        const device = try M.Device.init();
        errdefer device.deinit();
        const queue = try device.queue();
        p.commands = M.CommandSession.init(a, queue);
        p.storage = .{ .device = device, .queue = queue, .commands = &p.commands };
        return p;
    }
    pub fn room(p: *Self, _: std.mem.Allocator, _: std.Io) !u64 {
        // Stay within the GPU recommendation, with the same fixed driver/staging reserve as other native engines.
        const ceiling = p.storage.device.maxWorkingSet();
        if (ceiling == 0) return error.Qwen4MemoryBudget;
        return ceiling -| p.storage.device.allocated() -| @import("sys").memory_policy.margin;
    }
    const Loader = struct {
        pub fn load(_: @This(), a: std.mem.Allocator, b: *B, c: *const F.Config, problem: *F.Problem) !G {
            return G.load(a, b, c, problem, .hf);
        }
    };
    pub fn loader(_: *Self) Loader {
        return .{};
    }
    pub fn drawer(p: *Self, a: std.mem.Allocator, vocab: usize, rows: usize, streams: usize) !Drawer {
        return Drawer.load(a, &p.storage, vocab, rows, streams);
    }
    pub fn healthy(_: *const Self) bool {
        return true;
    }
    pub fn close(p: *Self) !void {
        try p.commands.deinit();
        p.storage.queue.deinit();
        p.storage.device.deinit();
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
    var p: F.Problem = .{};
    return Native.open(gpa, io, o, &p) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "Qwen4 Metal refused {s}: {s}", .{ if (p.name().len > 0) p.name() else "native load", @errorName(err) });
        return null;
    };
}
