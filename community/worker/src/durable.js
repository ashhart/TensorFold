import { DurableObject } from 'cloudflare:workers';
import migration from '../migrations/0001.sql';
import { handleRequest } from './index.js';
import { SqlDatabase } from './sqlite.js';

export class BenchmarkStore extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    const sql = ctx.storage.sql;
    ctx.storage.transactionSync(() => {
      const exists = sql.exec("SELECT name FROM sqlite_master WHERE type='table' AND name='publishers'").toArray();
      if (!exists.length) sql.exec(migration);
    });
    this.database = new SqlDatabase(sql);
  }
  async fetch(request) {
    const response = await handleRequest(request, { ...this.env, BENCHMARKS_DB: this.database });
    return response || new Response('Not found', { status: 404 });
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/benchmarks/v1/') || url.pathname === '/api/benchmarks/v1') {
      const id = env.BENCHMARKS_STORE.idFromName('community-v1');
      return env.BENCHMARKS_STORE.get(id).fetch(request);
    }
    const response = await handleRequest(request, env);
    if (response) return response;
    return env.ASSETS ? env.ASSETS.fetch(request) : new Response('Not found', { status: 404 });
  },
};
