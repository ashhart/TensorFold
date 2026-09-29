import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { SqlDatabase } from '../src/sqlite.js';

test('SQLite adapter binds values, returns first/all rows and accurate change counts', async () => {
  const db = new DatabaseSync(':memory:');
  db.exec('CREATE TABLE test_rows(id INTEGER PRIMARY KEY, value TEXT)');
  const adapter = new SqlDatabase({ exec(sql, ...values) {
    const statement = db.prepare(sql);
    return { toArray() {
      if (/^\s*SELECT/i.test(sql)) return statement.all(...values);
      statement.run(...values); return [];
    } };
  } });
  assert.equal((await adapter.prepare('INSERT INTO test_rows(value) VALUES (?)').bind('safe value').run()).meta.changes, 1);
  assert.equal((await adapter.prepare('SELECT * FROM test_rows WHERE id = ?').bind(1).first()).value, 'safe value');
  assert.equal(await adapter.prepare('SELECT value FROM test_rows WHERE id = ?').bind(1).first('value'), 'safe value');
  assert.equal(await adapter.prepare('SELECT * FROM test_rows WHERE id = ?').bind(2).first(), null);
  assert.equal((await adapter.prepare('SELECT * FROM test_rows').all()).results.length, 1);
  assert.equal((await adapter.prepare('UPDATE test_rows SET value = ? WHERE id = ?').bind('second', 9).run()).meta.changes, 0);
  db.close();
});
