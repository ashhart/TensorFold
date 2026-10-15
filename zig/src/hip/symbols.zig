//! One symbol-resolution path for admission fixtures and the full runtime.
const std = @import("std");
const platform = @import("sys");

pub fn resolve(comptime Api: type, lib: *platform.DynLib) error{MissingSymbol}!Api {
    var api: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
    }
    return api;
}
