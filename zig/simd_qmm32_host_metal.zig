//! Recording Metal facade for projection launch tests; no shader or device arithmetic.
const std = @import("std");
pub var limit: usize = 1024;
pub const objc = struct {
    pub const Pool = struct {
        pub fn push() Pool {
            return .{};
        }
        pub fn pop(_: Pool) void {}
    };
};
pub const CompileOptions = struct {
    pub fn mlx() CompileOptions {
        return .{};
    }
};
pub const Device = struct {
    pub fn generation(_: Device) u32 {
        return 15;
    }
};
pub const Library = struct {
    text: []const u8,
    pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !Library {
        return .{ .text = text };
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    name: [48]u8 = @splat(0),
    len: usize = 0,
    most: usize = 0,
    pub fn init(_: Device, lib: Library, name: []const u8, _: bool) !Pipeline {
        // Check every new specialization exists in the actual source consumed by the loader.
        if (std.mem.endsWith(u8, name, "_4") and std.mem.indexOf(u8, name, "mma_") != null) {
            var args: [40]u8 = undefined;
            const prefix = if (std.mem.startsWith(u8, name, "tf_sq_mma_")) "tf_sq_mma_" else "tf_sqw_mma_q4_";
            var it = std.mem.splitScalar(u8, name[prefix.len..], '_');
            const s = try std.fmt.parseInt(u32, it.next().?, 10);
            const nt = try std.fmt.parseInt(u32, it.next().?, 10);
            const needle = if (prefix.len == "tf_sq_mma_".len)
                try std.fmt.bufPrint(&args, "TF_SQ_MMA({d}, {d}, 4)", .{ s, nt })
            else
                try std.fmt.bufPrint(&args, "TF_SQW(4, {d}, {d}, 4)", .{ s, nt });
            if (std.mem.indexOf(u8, lib.text, needle) == null) return error.MissingTile;
        }
        var p: Pipeline = .{ .len = name.len, .most = limit };
        @memcpy(p.name[0..name.len], name);
        return p;
    }
    pub fn deinit(_: Pipeline) void {}
    pub fn maxThreads(p: Pipeline) usize {
        return p.most;
    }
    pub fn symbol(p: *const Pipeline) []const u8 {
        return p.name[0..p.len];
    }
};
pub const Size = struct { width: usize = 1, height: usize = 1, depth: usize = 1 };
pub const Buffer = struct { id: usize };
pub const Record = struct {
    offsets: [6]usize = @splat(0),
    ld: usize = 0,
    rows: usize = 0,
    groups: Size = .{},
    threads: usize = 0,
};
pub const ComputeEncoder = struct {
    record: *Record,
    pub fn setPipeline(_: ComputeEncoder, _: Pipeline) void {}
    pub fn setBuffer(e: ComputeEncoder, _: Buffer, off: usize, index: usize) void {
        e.record.offsets[index] = off;
    }
    pub fn setValue(e: ComputeEncoder, args: anytype, _: usize) void {
        e.record.rows = @intCast(args.r);
        e.record.ld = if (@hasField(@TypeOf(args), "ld")) @intCast(args.ld) else @intCast(args.n);
    }
    pub fn dispatchGroups(e: ComputeEncoder, groups: Size, threads: Size) void {
        e.record.groups = groups;
        e.record.threads = threads.width;
    }
};
