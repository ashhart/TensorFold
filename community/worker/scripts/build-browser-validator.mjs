import { writeFileSync } from 'node:fs';
import { BROWSER_PROTOCOL } from '../src/protocol.js';
writeFileSync(new URL('../src/browser-protocol.js', import.meta.url), 'export const BROWSER_PROTOCOL = ' + JSON.stringify(BROWSER_PROTOCOL) + ';\n');
