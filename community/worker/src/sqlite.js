// Adapt SQLite storage to the prepared queries used by the receiver.
export class SqlDatabase {
  constructor(sql) { this.sql = sql; }
  prepare(query) { return new Statement(this.sql, query, []); }
}
class Statement {
  constructor(sql, query, values) { this.sql = sql; this.query = query; this.values = values; }
  bind(...values) { return new Statement(this.sql, this.query, values); }
  async first(column) {
    const rows = this.sql.exec(this.query, ...this.values).toArray();
    const first = rows[0];
    return first ? (column === undefined ? first : first[column]) : null;
  }
  async all() { return { success: true, results: this.sql.exec(this.query, ...this.values).toArray() }; }
  async run() {
    this.sql.exec(this.query, ...this.values).toArray();
    const changes = this.sql.exec('SELECT changes() AS count').toArray()[0]?.count || 0;
    return { success: true, meta: { changes } };
  }
}
