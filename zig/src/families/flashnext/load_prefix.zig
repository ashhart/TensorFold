//! The loader's initialized prefix owns every successful acquisition until construction returns.
pub fn build(comptime Result: type, run: anytype, context: anytype, comptime construct: anytype) !Result {
    errdefer run.deinit();
    return construct(context, run);
}
