// Live updates for the Starlight operations dashboard; the page works without this script.
(function () {
  "use strict";
  var root = document.querySelector("[data-live-url]");
  if (!root || !window.fetch) { return; }
  var url = root.getAttribute("data-live-url");
  var last = 0;

  function rate(v) {
    if (v >= 1000) { return (v / 1000).toFixed(1) + "k"; }
    if (v >= 100) { return v.toFixed(0); }
    return v.toFixed(1);
  }
  function count(n) { return Number(n || 0).toLocaleString("en-US"); }
  function percent(v) { return Math.round((v || 0) * 100) + "%"; }
  function age(s) {
    if (s < 60) { return s + "s"; }
    if (s < 3600) { return Math.floor(s / 60) + "m " + (s % 60) + "s"; }
    return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
  }
  function ago(iso) {
    if (!iso) { return "never"; }
    var d = (Date.now() - Date.parse(iso)) / 1000;
    if (d < 60) { return "just now"; }
    if (d < 3600) { return Math.floor(d / 60) + " min ago"; }
    if (d < 172800) { return Math.floor(d / 3600) + " h ago"; }
    return Math.floor(d / 86400) + " days ago";
  }
  function set(key, value) {
    document.querySelectorAll('[data-live="' + key + '"]').forEach(function (el) { el.textContent = value; });
  }
  function points(values, w, h) {
    if (!values || !values.length) { return ""; }
    var top = Math.max.apply(null, values.concat([0]));
    var step = w / Math.max(1, values.length - 1);
    return values.map(function (v, i) {
      var y = top > 0 ? h - 1 - (v / top) * (h - 2) : h - 1;
      return (i * step).toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
  }
  function box(el) {
    var vb = el.ownerSVGElement ? el.ownerSVGElement.getAttribute("viewBox") : "0 0 120 32";
    var p = vb.split(" ");
    return { w: Number(p[2]), h: Number(p[3]) };
  }
  function plot(attr, areaAttr, name, values) {
    document.querySelectorAll("[" + attr + '="' + name + '"]').forEach(function (el) {
      var b = box(el);
      el.setAttribute("points", points(values, b.w, b.h));
    });
    document.querySelectorAll("[" + areaAttr + '="' + name + '"]').forEach(function (el) {
      var b = box(el);
      var line = points(values, b.w, b.h);
      el.setAttribute("points", line ? "0," + b.h.toFixed(1) + " " + line + " " + b.w.toFixed(1) + "," + b.h.toFixed(1) : "");
    });
  }
  function cell(tr, text, cls) {
    var td = document.createElement("td");
    if (cls) { td.className = cls; }
    if (text instanceof Node) { td.appendChild(text); } else { td.textContent = text; }
    tr.appendChild(td);
    return td;
  }
  function pill(state) {
    var s = document.createElement("span");
    s.className = "pill pill-" + state;
    s.textContent = state;
    return s;
  }
  function rows(name, list, colspan, empty, build) {
    var body = document.querySelector('[data-live-table="' + name + '"]');
    if (!body) { return; }
    while (body.firstChild) { body.removeChild(body.firstChild); }
    if (!list || !list.length) {
      var tr = document.createElement("tr");
      tr.className = "empty";
      cell(tr, empty).colSpan = colspan;
      body.appendChild(tr);
      return;
    }
    list.forEach(function (row) {
      var tr = document.createElement("tr");
      build(tr, row);
      body.appendChild(tr);
    });
  }
  function inFlight(tr, f) {
    cell(tr, f.model, "mono");
    cell(tr, f.who);
    cell(tr, f.project);
    var where = document.createElement("span");
    var route = document.createElement("span");
    route.className = "route route-" + f.route;
    route.textContent = f.route;
    where.appendChild(route);
    where.appendChild(document.createTextNode(" " + f.target));
    cell(tr, where);
    cell(tr, f.priority);
    cell(tr, f.client, "mono small");
    cell(tr, age(f.age_seconds), "num");
  }
  function connected(tr, c) {
    var who = document.createElement("span");
    who.textContent = c.who;
    if (c.who_type === "service_account") {
      var tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "service";
      who.appendChild(document.createTextNode(" "));
      who.appendChild(tag);
    }
    cell(tr, who);
    cell(tr, String(c.in_flight), "num");
    cell(tr, String(c.requests), "num");
    cell(tr, count(c.tokens), "num");
    cell(tr, c.last_client, "mono small");
    cell(tr, ago(c.last_seen));
  }
  function setCol(el, col, value) {
    var target = el.querySelector('[data-col="' + col + '"]');
    if (target) { target.textContent = value; }
    return target;
  }
  function nodes(list) {
    (list || []).forEach(function (n) {
      document.querySelectorAll('[data-node-id="' + n.id + '"]').forEach(function (el) {
        var st = el.querySelector('[data-col="state"]');
        if (st) { st.replaceChildren(pill(n.state)); }
        setCol(el, "ready", n.ready + " / " + n.replicas);
        setCol(el, "tps", n.fresh ? rate(n.tokens_per_second) : "—");
        setCol(el, "rw", n.fresh ? n.running + " · " + n.waiting : "—");
        setCol(el, "seen", ago(n.last_seen));
        var m = el.querySelector('[data-col="kv"] meter');
        if (m) { m.value = n.kv_usage; m.textContent = percent(n.kv_usage); }
      });
    });
  }
  function models(list) {
    (list || []).forEach(function (m) {
      document.querySelectorAll('[data-model-id="' + m.id + '"]').forEach(function (el) {
        var av = el.querySelector('[data-col="availability"]');
        if (av) { av.replaceChildren(pill(m.availability)); }
        setCol(el, "ready", m.ready + " / " + m.replicas);
        setCol(el, "nodes", String(m.nodes));
        setCol(el, "tps", rate(m.tokens_per_second));
        setCol(el, "queued", String(m.queued));
      });
    });
  }
  function warnings(list) {
    var ul = document.querySelector('[data-live-list="warnings"]');
    if (!ul) { return; }
    while (ul.firstChild) { ul.removeChild(ul.firstChild); }
    (list || []).forEach(function (w) {
      var li = document.createElement("li");
      li.textContent = w;
      ul.appendChild(li);
    });
  }
  function render(s) {
    if (s.tiles) {
      var t = s.tiles;
      set("tokens_per_second", rate(t.tokens_per_second));
      set("prompt_tokens_per_second", rate(t.prompt_tokens_per_second));
      set("in_flight", String(t.in_flight_local + t.in_flight_cloud));
      set("in_flight_local", String(t.in_flight_local));
      set("in_flight_cloud", String(t.in_flight_cloud));
      set("queued", String(t.queued));
      set("active_users", String(t.active_users));
      set("nodes_online", String(t.nodes_online));
      set("nodes_total", String(t.nodes_total));
      set("models_available", String(t.models_available));
      set("models_total", String(t.models_total));
      set("tokens_today", count(t.tokens_today));
      set("cost_today", t.cost_today);
      set("cloud_share", percent(t.cloud_share_today));
      warnings(t.warnings);
    }
    if (s.throughput) {
      plot("data-spark", "data-spark-area", "throughput", s.throughput);
      plot("data-chart", "data-chart-area", "throughput", s.throughput);
    }
    if (s.requests) {
      plot("data-spark", "data-spark-area", "requests", s.requests);
      plot("data-chart", "data-chart-area", "requests", s.requests);
    }
    rows("in_flight", s.in_flight, 7, "Nothing is running right now.", inFlight);
    rows("connected", s.connected, 6, "Nobody has used Starlight in the last 15 minutes.", connected);
    set("in_flight_rows", String((s.in_flight || []).length));
    set("connected_rows", String((s.connected || []).length));
    nodes(s.nodes);
    models(s.models);
    if (s.personal) {
      set("p_in_flight", String(s.personal.in_flight));
      set("p_requests_today", String(s.personal.requests_today));
      set("p_tokens_today", count(s.personal.tokens_today));
    }
    set("updated", "Live · updated " + new Date(s.at).toLocaleTimeString());
    root.classList.remove("stale");
  }
  function tick() {
    if (document.hidden) { return; }
    fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" }, cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (s) { last = Date.now(); render(s); })
      .catch(function () {
        if (Date.now() - last > 15000) {
          root.classList.add("stale");
          set("updated", "Reconnecting…");
        }
      });
  }
  setInterval(tick, 3000);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) { tick(); } });
  last = Date.now();
})();
