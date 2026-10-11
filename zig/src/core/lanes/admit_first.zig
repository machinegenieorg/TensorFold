//! Commit and hand off first tokens before a capable backend prepares its initial proposals.
const std = @import("std");
const be = @import("backend.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const trail = @import("trail.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Row = @import("logprob.zig").Row;
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// No batch-wide failure is returned after the first handoff; later failures belong to still-live streams.
pub fn streams(e: *Engine, list: []const *Stream, errs: []?anyerror, hook: ?be.Committed) !void {
    _ = e.arena.reset(.retain_capacity);
    const a = e.arena.allocator();
    const requests = try a.alloc(be.DraftRequest, list.len);
    const indices = try a.alloc(usize, list.len);
    try e.live.ensureUnusedCapacity(e.gpa, list.len);
    @memset(errs, null);
    for (list) |s| try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    if (list.len > 1 and e.backend.vtable.prefill_many != null) {
        try e.backend.vtable.prefill_many.?(e.backend.ptr, list, errs);
    } else for (list, errs) |s, *err| e.backend.prefill(s) catch |x| {
        err.* = x;
    };
    var n: usize = 0;
    for (list, errs, 0..) |s, *err, i| {
        if (err.*) |x| {
            if (x == error.Cancelled) e.release(s);
            continue;
        }
        const depth = first(e, s) catch |x| {
            err.* = x;
            if (x == error.Cancelled or s.finished) e.release(s);
            continue;
        };
        if (hook) |h| h.call(h.ptr, i);
        if (s.finished) continue;
        if (s.isCancelled()) {
            e.release(s);
            err.* = error.Cancelled;
            continue;
        }
        if (depth) |d| {
            requests[n] = .{ .stream = s, .follow = &.{}, .first = .{ .value = s.pending.? }, .rows = null, .start = s.prompt_len, .position = s.prompt_len + 1, .depth = d };
            indices[n] = i;
            n += 1;
        }
    }
    var kept: usize = 0;
    for (requests[0..n], indices[0..n]) |r, i| {
        if (r.stream.isCancelled()) {
            e.release(r.stream);
            errs[i] = error.Cancelled;
            continue;
        }
        requests[kept] = r;
        indices[kept] = i;
        kept += 1;
    }
    n = kept;
    if (n > 0) {
        e.backend.draft(requests[0..n]) catch |x| {
            for (indices[0..n]) |i| {
                errs[i] = x;
                if (x == error.Cancelled) e.release(list[i]);
            }
        };
        for (requests[0..n], indices[0..n]) |r, i| {
            if (errs[i] != null) continue;
            ready(e, r) catch |x| {
                errs[i] = x;
                if (x == error.Cancelled) e.release(r.stream);
            };
        }
    }
    for (list, errs) |s, err| {
        if (err == null and !s.finished) e.live.appendAssumeCapacity(s);
    }
}

/// Draw and commit exactly the selected first token, computing depth against the precommit budget.
fn first(e: *Engine, s: *Stream) !?u32 {
    if (s.isCancelled()) return error.Cancelled;
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.rows.clearRetainingCapacity();
    s.pending = null;
    s.cache_len = s.prompt_len;
    const position: u64 = s.prompt_len;
    const handle = try e.backend.first(s, position);
    var feed: be.Feed = .{ .handle = handle };
    if (try e.forcedNext(s)) |token| feed = .{ .value = token };
    const depth: ?u32 = if (e.cfg.family_mtp and s.drafts) @intCast(try e.rule.depth(win.who(s))) else null;
    const value = try e.readFeed(feed);
    if (e.log != null) {
        const drawn = if (feed == .handle) value else try e.backend.read(handle);
        try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(drawn)), f("token", int(value)) });
    }
    var row: [1]Row = undefined;
    if (s.logprobs != null) row[0] = (try e.backend.firstRow(s)).forToken(value);
    _ = try s.commit(e.gpa, &.{value}, if (s.logprobs != null) &row else &.{});
    s.pending = value;
    if (!s.finished and depth == null and e.cfg.pipelined and s.logprobs == null) try e.queueNext(s, feed);
    try trail.resolve(e);
    if (s.finished) {
        try trail.finish(e, s);
        e.release(s);
    }
    return depth;
}

/// Make successful initial proposals visible to the next verifier round, before admission returns.
fn ready(e: *Engine, r: be.DraftRequest) !void {
    const s = r.stream;
    if (s.isCancelled()) return error.Cancelled;
    s.dropHeld(e.gpa);
    s.next = .{ .count = r.depth };
    if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
        s.next = held;
    };
    try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(r.depth)), f("position", int(r.position)), f("follow", .{ .u32s = &.{s.pending.?} }), f("rows", .null) });
}
