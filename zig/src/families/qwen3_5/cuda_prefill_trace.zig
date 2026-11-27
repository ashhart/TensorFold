//! Setup-time adapters select traced prompt specializations; the ordinary backend retains its original pointers.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const dr = lanes.drafter;
const be = lanes.backend;
const trace = cuda.prefill_attribution;
const prefill = @import("cuda_prefill.zig");
const readout = @import("cuda_readout.zig");
pub const Ticket = trace.Ticket;
const stages = @import("cuda_prefill_stages.zig");
pub const open = stages.open;
const begin = stages.begin;
const end = stages.end;
const failed = stages.failed;
fn enter(self: anytype, stream: *lanes.Stream) bool {
    const t = trace.owner orelse return false;
    return t.enter(stream, stream.prompt().len, self.e.scratch_capacity, if (self.e.scratch_buf) |b| b.len else 0);
}
pub fn wrapBackend(comptime L: type, original: be.Backend) be.Backend {
    if (trace.owner == null) return original;
    const A = Adapter(L);
    A.original = original.vtable;
    A.table = original.vtable.*;
    A.table.prefill = A.whole;
    A.table.prefill_step = A.step;
    A.table.release = A.release;
    var wrapped = original;
    wrapped.vtable = &A.table;
    return wrapped;
}
fn Adapter(comptime L: type) type {
    return struct {
        var original: *const be.Backend.VTable = undefined;
        var table: be.Backend.VTable = undefined;
        fn cast(p: *anyopaque) *L {
            return @ptrCast(@alignCast(p));
        }
        fn whole(p: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = cast(p);
            if (!enter(self, s)) return original.prefill(p, s);
            const t = trace.owner.?;
            defer t.leave();
            const ticket = begin(.request, true);
            defer end(ticket);
            errdefer failed(ticket);
            const scratch = begin(.scratch, false);
            end(scratch);
            if (s.readout != null) {
                while (!try readout.step(self, s, 0)) {}
                return;
            }
            return prefill.runMode(true, self, s);
        }
        fn step(p: *anyopaque, s: *lanes.Stream, rows: u32) anyerror!bool {
            const self = cast(p);
            if (!enter(self, s)) return original.prefill_step.?(p, s, rows);
            const t = trace.owner.?;
            defer t.leave();
            const ticket = begin(.request, true);
            defer end(ticket);
            errdefer failed(ticket);
            const scratch = begin(.scratch, false);
            end(scratch);
            if (s.readout != null) return readout.step(self, s, rows);
            return prefill.stepMode(true, self, s, rows);
        }
        fn finish(s: *lanes.Stream) void {
            if (trace.owner) |t| if (t.request == @as(*anyopaque, s)) {
                t.finished = true;
                t.collect(false);
            };
        }
        fn release(p: *anyopaque, s: *lanes.Stream) void {
            original.release(p, s);
            finish(s);
        }
    };
}
const Node = struct { ptr: *anyopaque, original: *const dr.Drafter.VTable, table: dr.Drafter.VTable };
var nodes: std.ArrayList(*Node) = .empty;
pub fn wrapDrafter(d: anytype, original: dr.Drafter) dr.Drafter {
    const t = trace.owner orelse return original;
    if (comptime !@hasField(@TypeOf(d.*), "ring_fns")) {
        t.tap_functions[0] = d.fns.gather_taps.handle;
        t.tap_functions[1] = d.fns.gather_rows.handle;
    } else {
        t.tap_functions[2] = d.ring_fns.gather_taps.handle;
        t.tap_functions[3] = d.ring_fns.gather_rows.handle;
    }
    for (nodes.items) |node| if (node.ptr == original.ptr) {
        node.original = original.vtable;
        node.table = original.vtable.*;
        node.table.open = draftOpen;
        node.table.absorb = absorb;
        var wrapped = original;
        wrapped.vtable = &node.table;
        return wrapped;
    };
    const node = t.a.create(Node) catch {
        t.dropped += 1;
        return original;
    };
    node.* = .{ .ptr = original.ptr, .original = original.vtable, .table = original.vtable.* };
    nodes.append(t.a, node) catch {
        t.a.destroy(node);
        t.dropped += 1;
        return original;
    };
    node.table.open = draftOpen;
    node.table.absorb = absorb;
    var wrapped = original;
    wrapped.vtable = &node.table;
    return wrapped;
}
fn nodeOf(p: *anyopaque) *Node {
    for (nodes.items) |node| if (node.ptr == p) return node;
    unreachable;
}
fn draftEnter(s: *lanes.Stream) bool {
    const t = trace.owner orelse return false;
    if (t.request != @as(*anyopaque, s)) return false;
    return t.enter(s, t.prompt_tokens, t.scratch_rows, t.scratch_bytes);
}
fn draftOpen(p: *anyopaque, s: *lanes.Stream) anyerror!void {
    const node = nodeOf(p);
    if (!draftEnter(s)) return node.original.open(p, s);
    defer trace.owner.?.leave();
    const ticket = begin(.drafter_open, true);
    defer end(ticket);
    errdefer failed(ticket);
    return node.original.open(p, s);
}
fn absorb(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
    const node = nodeOf(p);
    var selected: ?dr.Absorb = null;
    for (items) |item| if (item.prompt and trace.owner.?.request == @as(*anyopaque, item.stream)) {
        selected = item;
        break;
    };
    const item = selected orelse return node.original.absorb(p, items);
    if (!draftEnter(item.stream)) return node.original.absorb(p, items);
    const t = trace.owner.?;
    defer t.leave();
    const trailing = item.start == item.stream.prompt().len -| 1 and item.features.rows == 1;
    const ticket = t.begin(.{ .stage = if (trailing) .trailing_absorb else .drafter_absorb, .start_token = item.start, .rows = item.features.rows, .operation = "tap_gather_and_absorb" }, t.stream, true);
    defer end(ticket);
    errdefer failed(ticket);
    return node.original.absorb(p, items);
}
pub fn close() void {
    if (trace.owner) |t| {
        for (nodes.items) |node| t.a.destroy(node);
        nodes.deinit(t.a);
        nodes = .empty;
    }
    trace.Trace.close();
}
