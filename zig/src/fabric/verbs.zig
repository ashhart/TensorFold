//! The system's verbs library by dlopen (Apple's librdma, Linux libibverbs): every entry point resolved by name.
const std = @import("std");
const builtin = @import("builtin");
const abi = @import("verbs_abi.zig");

pub const Error = error{ LibraryUnavailable, MissingSymbol, NoDeviceList, NoSuchDevice, VerbsFailed };

/// macOS 26.2 and later ship librdma (Thunderbolt RDMA, and MCDMA's provider when installed); Linux has rdma-core.
pub const default_path: [:0]const u8 = if (builtin.os.tag == .macos) "/usr/lib/librdma.dylib" else "libibverbs.so.1";

const Ctx = *abi.Context;

/// Every field is resolved by its exact name; a missing one refuses the library.
pub const Api = struct {
    ibv_get_device_list: *const fn (num: *c_int) callconv(.c) ?[*:null]?*abi.Device,
    ibv_free_device_list: *const fn (list: [*:null]?*abi.Device) callconv(.c) void,
    ibv_get_device_name: *const fn (dev: *abi.Device) callconv(.c) ?[*:0]const u8,
    ibv_open_device: *const fn (dev: *abi.Device) callconv(.c) ?Ctx,
    ibv_close_device: *const fn (ctx: Ctx) callconv(.c) c_int,
    ibv_query_device: *const fn (ctx: Ctx, attr: *abi.DeviceAttr) callconv(.c) c_int,
    ibv_query_port: *const fn (ctx: Ctx, port: u8, attr: *abi.PortAttr) callconv(.c) c_int,
    ibv_query_gid: *const fn (ctx: Ctx, port: u8, index: c_int, gid: *abi.Gid) callconv(.c) c_int,
    ibv_alloc_pd: *const fn (ctx: Ctx) callconv(.c) ?*abi.Pd,
    ibv_dealloc_pd: *const fn (pd: *abi.Pd) callconv(.c) c_int,
    ibv_reg_mr: *const fn (pd: *abi.Pd, addr: ?*anyopaque, len: usize, access: c_int) callconv(.c) ?*abi.Mr,
    ibv_dereg_mr: *const fn (mr: *abi.Mr) callconv(.c) c_int,
    ibv_create_cq: *const fn (ctx: Ctx, cqe: c_int, user: ?*anyopaque, channel: ?*anyopaque, vector: c_int) callconv(.c) ?*abi.Cq,
    ibv_destroy_cq: *const fn (cq: *abi.Cq) callconv(.c) c_int,
    ibv_create_qp: *const fn (pd: *abi.Pd, attr: *abi.QpInitAttr) callconv(.c) ?*abi.Qp,
    ibv_modify_qp: *const fn (qp: *abi.Qp, attr: *abi.QpAttr, mask: c_int) callconv(.c) c_int,
    ibv_destroy_qp: *const fn (qp: *abi.Qp) callconv(.c) c_int,
    ibv_wc_status_str: *const fn (status: c_int) callconv(.c) [*:0]const u8,
};

/// Exported by rdma-core 34+ but not in every header; used only when present.
const GidEx = *const fn (ctx: Ctx, port: u32, index: u32, entry: *abi.GidEntry, flags: u32, size: usize) callconv(.c) c_int;
const InOrder = *const fn (qp: *abi.Qp, op: c_int, flags: u32) callconv(.c) c_int;

pub const Verbs = struct {
    lib: std.DynLib,
    api: Api,
    gid_ex: ?GidEx = null,
    in_order: ?InOrder = null,

    pub fn open(path: ?[:0]const u8) Error!Verbs {
        var lib = std.DynLib.openZ((path orelse default_path).ptr) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        var v: Verbs = .{ .lib = lib, .api = api };
        v.gid_ex = v.lib.lookup(GidEx, "_ibv_query_gid_ex");
        v.in_order = v.lib.lookup(InOrder, "ibv_query_qp_data_in_order");
        return v;
    }

    pub fn close(v: *Verbs) void {
        v.lib.close();
    }

    /// Copy every device name into `names` (newline-separated) and free the list; opens no device.
    pub fn deviceNames(v: *const Verbs, names: *std.Io.Writer) (Error || std.Io.Writer.Error)!usize {
        var n: c_int = 0;
        const list = v.api.ibv_get_device_list(&n) orelse return error.NoDeviceList;
        defer v.api.ibv_free_device_list(list);
        var count: usize = 0;
        while (list[count]) |dev| : (count += 1) {
            const name = v.api.ibv_get_device_name(dev) orelse continue;
            try names.print("{s}\n", .{std.mem.span(name)});
        }
        return count;
    }

    /// Open the device called `name`; the list is freed before returning, as verbs allows once opened.
    pub fn openDevice(v: *const Verbs, name: []const u8) Error!Ctx {
        var n: c_int = 0;
        const list = v.api.ibv_get_device_list(&n) orelse return error.NoDeviceList;
        defer v.api.ibv_free_device_list(list);
        var i: usize = 0;
        while (list[i]) |dev| : (i += 1) {
            const got = v.api.ibv_get_device_name(dev) orelse continue;
            if (std.mem.eql(u8, std.mem.span(got), name)) return v.api.ibv_open_device(dev) orelse error.VerbsFailed;
        }
        return error.NoSuchDevice;
    }

    pub fn port(v: *const Verbs, ctx: Ctx, num: u8) Error!abi.PortAttr {
        var attr = std.mem.zeroes(abi.PortAttr);
        if (v.api.ibv_query_port(ctx, num, &attr) != 0) return error.VerbsFailed;
        return attr;
    }

    pub fn device(v: *const Verbs, ctx: Ctx) Error!abi.DeviceAttr {
        var attr = std.mem.zeroes(abi.DeviceAttr);
        if (v.api.ibv_query_device(ctx, &attr) != 0) return error.VerbsFailed;
        return attr;
    }

    pub fn gid(v: *const Verbs, ctx: Ctx, num: u8, index: u32) Error!abi.Gid {
        var g: abi.Gid = .{ .raw = @splat(0) };
        if (v.api.ibv_query_gid(ctx, num, @intCast(index), &g) != 0) return error.VerbsFailed;
        return g;
    }

    /// The GID's type (0 IB, 1 RoCE v1, 2 RoCE v2), or null when the library cannot say.
    pub fn gidType(v: *const Verbs, ctx: Ctx, num: u8, index: u32) ?u32 {
        const f = v.gid_ex orelse return null;
        var e = std.mem.zeroes(abi.GidEntry);
        if (f(ctx, num, index, &e, 0, @sizeOf(abi.GidEntry)) != 0) return null;
        return e.gid_type;
    }

    pub fn statusText(v: *const Verbs, status: c_int) []const u8 {
        return std.mem.span(v.api.ibv_wc_status_str(status));
    }
};

test "verbs smoke: open the device list and close it (TF_FABRIC_VERBS=1)" {
    const flag = std.c.getenv("TF_FABRIC_VERBS") orelse return error.SkipZigTest;
    if (flag[0] != '1') return error.SkipZigTest;
    var v = Verbs.open(null) catch |err| {
        std.debug.print("verbs: {s} unavailable ({s})\n", .{ default_path, @errorName(err) });
        return error.SkipZigTest;
    };
    defer v.close();
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    const n = try v.deviceNames(&out.writer);
    std.debug.print("verbs: {d} devices from {s}\n{s}", .{ n, default_path, out.written() });
}
