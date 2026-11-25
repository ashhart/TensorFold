//! Minimal clean-room Objective-C runtime bridge for direct Metal 4 calls on macOS arm64.
const std = @import("std");

pub const Id = *anyopaque;
pub const Class = *anyopaque;
pub const Sel = *anyopaque;

extern "c" fn objc_getClass(name: [*:0]const u8) ?Class;
extern "c" fn sel_registerName(name: [*:0]const u8) Sel;
extern "c" fn objc_msgSend() void;
extern "c" fn objc_autoreleasePoolPush() *anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: *anyopaque) void;

pub fn class(comptime name: [:0]const u8) Class {
    return objc_getClass(name.ptr) orelse @panic("Missing Objective-C class: " ++ name);
}

pub fn classOpt(comptime name: [:0]const u8) ?Class {
    return objc_getClass(name.ptr);
}

pub fn sel(comptime name: [:0]const u8) Sel {
    return sel_registerName(name.ptr);
}

pub fn responds(target: Id, comptime selector: [:0]const u8) bool {
    return send(bool, target, "respondsToSelector:", .{sel(selector)});
}

pub fn allocInitOpt(comptime cls_name: [:0]const u8) ?Id {
    const cls = classOpt(cls_name) orelse return null;
    const raw = send(?Id, cls, "alloc", .{}) orelse return null;
    return send(?Id, raw, "init", .{});
}

pub const Pool = struct {
    ptr: *anyopaque,

    pub fn push() Pool {
        return .{ .ptr = objc_autoreleasePoolPush() };
    }

    pub fn pop(self: Pool) void {
        objc_autoreleasePoolPop(self.ptr);
    }
};

fn FnType(comptime Ret: type, comptime ArgsTuple: type) type {
    const ftypes = @typeInfo(ArgsTuple).@"struct".field_types;
    var param_types: [ftypes.len + 2]type = undefined;
    param_types[0] = Id;
    param_types[1] = Sel;
    inline for (ftypes, 0..) |ft, i| {
        param_types[i + 2] = ft;
    }
    return @Fn(&param_types, &@splat(.{}), Ret, .{ .@"callconv" = .c });
}

/// Dispatch an Objective-C message on arm64 (where objc_msgSend handles all return types).
pub inline fn send(comptime Ret: type, target: Id, comptime selector: [:0]const u8, args: anytype) Ret {
    const F = FnType(Ret, @TypeOf(args));
    const func: *const F = @ptrCast(&objc_msgSend);
    return @call(.auto, func, .{ target, sel(selector) } ++ args);
}

pub fn retain(obj: Id) Id {
    return send(Id, obj, "retain", .{});
}

pub fn release(obj: Id) void {
    send(void, obj, "release", .{});
}

pub fn allocInit(comptime cls_name: [:0]const u8) Id {
    const raw = send(Id, class(cls_name), "alloc", .{});
    return send(Id, raw, "init", .{});
}

/// Create an owned NSString from a UTF-8 Zig slice. Caller must `release`.
pub fn nsString(bytes: []const u8) Id {
    const raw = send(Id, class("NSString"), "alloc", .{});
    // NSUTF8StringEncoding = 4
    return send(Id, raw, "initWithBytes:length:encoding:", .{ bytes.ptr, bytes.len, @as(usize, 4) });
}

/// Return a UTF-8 error description string (valid within active autorelease pool).
pub fn errorMessage(err: ?Id) []const u8 {
    const e = err orelse return "(unknown error)";
    const desc = send(?Id, e, "localizedDescription", .{}) orelse return "(no description)";
    const cstr = send(?[*:0]const u8, desc, "UTF8String", .{}) orelse return "(invalid utf8)";
    return std.mem.span(cstr);
}
