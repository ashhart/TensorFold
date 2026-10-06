//! First-party libmcdma-fabric ABI 1, loaded only from an explicit absolute library path.
const std = @import("std");

pub const Context = opaque {};
pub const Peer = opaque {};
pub const progress_thread: u32 = 1;
pub const thunderbolt: c_int = 2;
pub const Error = error{ LibraryUnavailable, MissingSymbol, WrongAbi, InvalidPath };

pub const Api = struct {
    abi: *const fn () callconv(.c) u32,
    open: *const fn ([*:0]const u8, c_int, c_int, *anyopaque, usize, c_int, u64, u32, *?*Context) callconv(.c) c_int,
    connect: *const fn (*Context, [*:0]const u8, c_int, c_int, [*:0]const u8, u64, *?*Peer) callconv(.c) c_int,
    link: *const fn (*const Peer) callconv(.c) c_int,
    peer_length: *const fn (*const Peer) callconv(.c) u64,
    write: *const fn (*Peer, u64, u64, u64) callconv(.c) c_int,
    signal: *const fn (*Peer, u64, u64) callconv(.c) c_int,
    flush: *const fn (*Peer, u64) callconv(.c) c_int,
    disconnect: *const fn (*?*Peer) callconv(.c) void,
    close: *const fn (*?*Context) callconv(.c) void,
};

/// Optional symbols: a write and its signal as one message (libraries that have it).
pub const WriteSignal = *const fn (*Peer, u64, u64, u64, u64, u64) callconv(.c) c_int;
pub const ConnectLinks = *const fn (*Context, [*:0]const u8, [*]const u16, [*]const u16, c_uint, [*:0]const u8, u64, *?*Peer) callconv(.c) c_int;
pub const MaxLinks = *const fn () callconv(.c) c_uint;
pub const ws_room: usize = 64;

pub const Library = struct {
    handle: std.DynLib,
    api: Api,
    write_signal: ?WriteSignal = null,
    connect_links: ?ConnectLinks = null,
    max_links: ?MaxLinks = null,

    /// No install-path or loader-search fallback; the caller supplies the inspected user-directory library.
    pub fn open(path: [:0]const u8) Error!Library {
        if (!std.fs.path.isAbsolute(path)) return error.InvalidPath;
        var handle = std.DynLib.openZ(path.ptr) catch return error.LibraryUnavailable;
        errdefer handle.close();
        var api: Api = undefined;
        inline for (@typeInfo(Api).@"struct".field_names) |name| {
            @field(api, name) = handle.lookup(@FieldType(Api, name), "mcdma_fabric_" ++ name) orelse return error.MissingSymbol;
        }
        if (api.abi() != 1) return error.WrongAbi;
        return .{ .handle = handle, .api = api, .write_signal = handle.lookup(WriteSignal, "mcdma_fabric_write_signal"), .connect_links = handle.lookup(ConnectLinks, "mcdma_fabric_connect_links"), .max_links = handle.lookup(MaxLinks, "mcdma_fabric_max_links") };
    }

    pub fn close(self: *Library) void {
        self.handle.close();
    }
};
