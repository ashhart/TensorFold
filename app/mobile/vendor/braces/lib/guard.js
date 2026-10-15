'use strict';
const MAX_DEPTH = 128;
const MAX_NODES = 65536;
function guard(ast) {
  const stack = [[ast, 0]];
  const seen = new Set();
  let count = 0;
  while (stack.length) {
    const [node, depth] = stack.pop();
    if (!node || typeof node !== 'object') throw new TypeError('Invalid brace AST node');
    if (depth > MAX_DEPTH || ++count > MAX_NODES) throw new RangeError('Brace AST exceeds safety limits');
    if (seen.has(node)) throw new RangeError('Brace AST contains a cycle or shared subtree');
    seen.add(node);
    if (node.nodes) {
      if (!Array.isArray(node.nodes)) throw new TypeError('Invalid brace AST children');
      for (const child of node.nodes) stack.push([child, depth + 1]);
    }
  }
}
guard.MAX_DEPTH = MAX_DEPTH;
module.exports = guard;
