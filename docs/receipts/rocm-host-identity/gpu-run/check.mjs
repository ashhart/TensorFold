import {validateReceipt} from './protocol.mjs'; import fs from 'fs';
for (const f of process.argv.slice(2)) {
  const r = JSON.parse(fs.readFileSync(f));
  for (const backend of ['rocm','cuda']) {
    const x = structuredClone(r); x.runtime.backend = backend;
    try { validateReceipt(x); console.log(f, backend, 'ACCEPTED'); } catch (e) { console.log(f, backend, 'REFUSED:', e.message); }
  }
}
