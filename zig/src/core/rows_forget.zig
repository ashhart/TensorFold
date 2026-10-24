//! Remove saved conversations on the engine thread, including writes that could restore an erased span.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const sync = @import("learned_sync.zig");

pub fn forget(r: *rows.Rows, span: []const u32, out: *pc.Forgotten) void {
    out.conversation_disk = true;
    if (r.forget_pending) out.failed += 1;
    const failures = out.failed;
    if (r.job) |j| if (carries(j.entry.tokens, span)) {
        rows.write.abortForErase(r, j) catch {
            out.failed += 1;
        };
    };
    var i = r.queue.items.len;
    while (i > 0) {
        i -= 1;
        const q = r.queue.items[i];
        if (!carries(q.entry.tokens, span)) continue;
        r.gpa.free(q.starts);
        _ = r.queue.orderedRemove(i);
    }
    std.crypto.secureZero(u8, r.stage);
    i = r.states.items.len;
    while (i > 0) {
        i -= 1;
        const state = &r.states.items[i];
        if (!pc.holdsSpan(state.tokens, span)) {
            if (pc.weakSpanTail(state.tokens, span)) out.held += 1;
            continue;
        }
        remove(r, i) catch {
            out.failed += 1;
            continue;
        };
        out.conversations += 1;
    }
    collect(r) catch {
        out.failed += 1;
    };
    syncDir(r.dir) catch {
        out.failed += 1;
    };
    if (out.failed > failures) r.forget_pending = true;
}

fn carries(tokens: []const u32, span: []const u32) bool {
    return pc.holdsSpan(tokens, span) or pc.weakSpanTail(tokens, span);
}

fn remove(r: *rows.Rows, index: usize) !void {
    const state = &r.states.items[index];
    var wants: std.ArrayList(rows.write.Want) = .empty;
    defer wants.deinit(r.gpa);
    try r.wantsOf(r.gpa, state, &wants);
    try unlink(r.dir, rows.chain.stem(state.name), ".st");
    r.ref(wants.items, -1);
    const gone = r.states.orderedRemove(index);
    std.crypto.secureZero(u32, gone.tokens);
    r.gpa.free(gone.tokens);
    r.gpa.free(gone.spans);
    r.gpa.free(gone.starts);
}

fn collect(r: *rows.Rows) !void {
    var gone: std.ArrayList(rows.chain.Stem) = .empty;
    defer gone.deinit(r.gpa);
    var it = r.blocks.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.refs == 0 and !rows.write.wanted(r, entry.key_ptr.*)) try gone.append(r.gpa, entry.key_ptr.*);
    }
    var failed = false;
    for (gone.items) |name| {
        unlink(r.dir, name, ".blk") catch {
            failed = true;
            continue;
        };
        _ = r.blocks.remove(name);
    }
    if (failed) return error.RowsErase;
}

fn unlink(dir: []const u8, name: rows.chain.Stem, ext: []const u8) !void {
    var buffer: [1200]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buffer, "{s}/{s}{s}", .{ dir, std.fmt.bytesToHex(name, .lower), ext }, 0);
    const rc = std.c.unlink(path);
    if (rc != 0 and std.c.errno(rc) != .NOENT) return error.RowsErase;
}

fn syncDir(dir: [:0]const u8) !void {
    const fd = std.c.open(dir, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.RowsErase;
    defer _ = std.c.close(fd);
    if (!sync.full(fd)) return error.RowsErase;
}
