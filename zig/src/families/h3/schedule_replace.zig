//! Replacement projects into temporary owners and publishes only after steps and command completion succeed.
pub fn run(model: anytype, op: anytype) !@typeInfo(@TypeOf(op.makeSteps())).error_union.payload {
    const steps = try op.makeSteps();
    errdefer op.freeSteps(steps);
    const Ref = @TypeOf(model.final_table);
    const tables = try op.allocator.alloc(Ref, model.blocks.len);
    defer op.allocator.free(tables);
    var made: usize = 0;
    errdefer for (tables[0..made]) |table| op.release(table);
    for (tables, 0..) |*table, i| {
        table.* = try op.block(i);
        made += 1;
    }
    const final = try op.final();
    errdefer op.release(final);
    try op.finish();
    for (tables, 0..) |table, i| {
        op.release(model.blocks[i].tables);
        model.blocks[i].tables = table;
    }
    op.release(model.final_table);
    model.final_table = final;
    model.adaln = op.adaln;
    model.times = op.times;
    model.lines = op.lines;
    return steps;
}
