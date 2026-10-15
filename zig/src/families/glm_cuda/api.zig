//! GLM-5.3-Flash on CUDA, two ranks: the weight store (Python's prepared-weight names) and kernels the family shares.

const std = @import("std");
const cuda = @import("cuda");

pub const DType = enum { bf16, f16, f32, i32, i64, u8, i8, i16, f8e4m3, bool_ };

pub fn dtypeBytes(t: DType) usize {
    return switch (t) {
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        .i64 => 8,
        .u8, .i8, .f8e4m3, .bool_ => 1,
    };
}

pub fn parseDType(s: []const u8) ?DType {
    const map = std.StaticStringMap(DType).initComptime(.{ .{ "bfloat16", .bf16 }, .{ "float16", .f16 }, .{ "float32", .f32 }, .{ "int32", .i32 }, .{ "int64", .i64 }, .{ "uint8", .u8 }, .{ "int8", .i8 }, .{ "int16", .i16 }, .{ "float8_e4m3fn", .f8e4m3 }, .{ "bool", .bool_ } });
    return map.get(s);
}

/// A device tensor: address, element type, shape (row-major, contiguous unless ``stride`` says otherwise).
pub const Tensor = struct {
    ptr: u64,
    dtype: DType,
    shape: [5]i64 = .{ 1, 1, 1, 1, 1 },
    nd: u8 = 0,

    pub fn numel(t: Tensor) usize {
        var n: usize = 1;
        for (t.shape[0..t.nd]) |d| n *= @intCast(d);
        return n;
    }
    pub fn bytes(t: Tensor) usize {
        return t.numel() * dtypeBytes(t.dtype);
    }
    pub fn dim(t: Tensor, i: usize) i64 {
        return t.shape[i];
    }
};

/// Prepared weights by name, all on the device. ``deinit`` frees what ``owned`` holds.
pub const Store = struct {
    gpa: std.mem.Allocator,
    map: std.StringHashMap(Tensor),
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Host-side facts the loader records (vocabulary offset, widths), by name.
    ints: std.StringHashMap(i64),
    /// Host data by name (owned).
    host: std.StringHashMap([]u8),

    pub fn init(gpa: std.mem.Allocator) Store {
        return .{ .gpa = gpa, .map = .init(gpa), .ints = .init(gpa), .host = .init(gpa) };
    }

    pub fn deinit(self: *Store) void {
        for (self.owned.items) |*b| b.free();
        self.owned.deinit(self.gpa);
        var it = self.map.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.map.deinit();
        var it2 = self.ints.keyIterator();
        while (it2.next()) |k| self.gpa.free(k.*);
        self.ints.deinit();
        var it3 = self.host.iterator();
        while (it3.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            self.gpa.free(e.value_ptr.*);
        }
        self.host.deinit();
    }

    pub fn put(self: *Store, name: []const u8, t: Tensor) !void {
        try self.map.put(try self.gpa.dupe(u8, name), t);
    }

    pub fn get(self: *const Store, name: []const u8) !Tensor {
        return self.map.get(name) orelse {
            std.log.err("glm: no weight named {s}", .{name});
            return error.MissingWeight;
        };
    }

    pub fn getf(self: *const Store, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [256]u8 = undefined;
        return self.get(try std.fmt.bufPrint(&buf, fmt, args));
    }
};

/// The device, its stream and the two ranks' communicator.
pub const Ctx = struct {
    d: *const cuda.Driver,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    nccl: *cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: u8,
    world: u8 = 2,
    /// Small all-gathers as RDMA writes over RoCE (cuda.roce), when both ranks brought it up; NCCL otherwise.
    roce: ?*cuda.roce.Gather = null,

    /// Both ranks' ``count`` values from ``send`` into ``recv`` in rank order (ncclAllGather's layout and bits).
    pub fn allGather(c: *const Ctx, send: u64, recv: u64, count: usize, dt: cuda.nccl.DataType, what: []const u8) !void {
        const size: usize = switch (dt) {
            .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
        if (c.roce) |g| if (g.fits(count * size, send, recv)) return g.gather(c.stream.handle, send, recv, count * size);
        try c.nccl.check(c.nccl.api.ncclAllGather(send, recv, count, dt, c.comm, c.stream.handle), what);
    }
};

/// The kernels: this family's modules and the shared ones it launches (embedded fatbins).
pub const Kernels = struct {
    ext: [ext_names.len]cuda.Module,

    pub const ext_names = [_][]const u8{ "fn_experts", "fn_experts_prefill", "glm_kda", "fn_sample", "glm_ops", "glm_experts", "glm_experts_prefill", "glm_vision", "glm_experts3", "glm_experts3_prefill", "glm_mm", "glm_rows", "glm_latent", "glm_index", "glm_exl3", "glm_dflash" };

    pub fn load(d: *const cuda.Driver) !Kernels {
        var k: Kernels = .{ .ext = undefined };
        const images = [_][]const u8{ cuda.kernels.fn_experts, cuda.kernels.fn_experts_prefill, cuda.kernels.glm_kda, cuda.kernels.fn_sample, cuda.kernels.glm_ops, cuda.kernels.glm_experts, cuda.kernels.glm_experts_prefill, cuda.kernels.glm_vision, cuda.kernels.glm_experts3, cuda.kernels.glm_experts3_prefill, cuda.kernels.glm_mm, cuda.kernels.glm_rows, cuda.kernels.glm_latent, cuda.kernels.glm_index, cuda.kernels.glm_exl3, cuda.kernels.glm_dflash };
        for (images, 0..) |img, i| k.ext[i] = try cuda.Module.load(d, img);
        return k;
    }

    pub fn deinit(self: *Kernels) void {
        for (&self.ext) |*m| m.unload();
    }
};
