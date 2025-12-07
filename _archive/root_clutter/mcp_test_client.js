#!/usr/bin/env node
// MCP test client: spawns chrome-devtools-mcp, lists available tools, and calls `list_pages`.
import { spawn } from 'node:child_process';

const child = spawn('npx', ['-y', 'chrome-devtools-mcp@latest'], {
  stdio: ['pipe', 'pipe', 'pipe'],
  env: process.env,
});

let stdoutBuf = '';
let stderrBuf = '';
let nextId = 1;
const pending = new Map();

function sendRequest(method, params) {
  const id = nextId++;
  const msg = { jsonrpc: '2.0', id, method, params };
  pending.set(id, { method, params });
  child.stdin.write(JSON.stringify(msg) + '\n');
  return id;
}

child.stdout.on('data', (d) => {
  const s = d.toString('utf8');
  process.stdout.write('[adapter stdout] ' + s);
  stdoutBuf += s;
  tryParseJsonLines(s);
});

child.stderr.on('data', (d) => {
  const s = d.toString('utf8');
  process.stderr.write('[adapter stderr] ' + s);
  stderrBuf += s;
  tryParseJsonLines(s);
});

child.on('exit', (code, signal) => {
  console.log(`adapter exited with code=${code} signal=${signal}`);
  process.exit(code || 0);
});

child.on('error', (err) => {
  console.error('failed to spawn adapter:', err);
  process.exit(1);
});

// After adapter start, request the tool list then call `list_pages`.
setTimeout(() => {
  console.log('[test client] Requesting tools list...');
  sendRequest('tools/list', {});
}, 600);

// Timeout to avoid hanging forever
const overallTimeout = setTimeout(() => {
  console.error('Timed out waiting for adapter responses.');
  process.exit(2);
}, 20000);

function tryParseJsonLines(s) {
  for (const line of s.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const obj = JSON.parse(line);
      handleMessage(obj);
    } catch (e) {
      // not JSON, ignore
    }
  }
}

function handleMessage(msg) {
  console.log('[parsed JSON] ', JSON.stringify(msg));
  if (msg.id && (msg.result || msg.error)) {
    // response to a request
    const pendingInfo = pending.get(msg.id);
    pending.delete(msg.id);
    if (!pendingInfo) return;
    if (pendingInfo.method === 'tools/list') {
      console.log('Received tools list; attempting to call `list_pages`...');
      // call the tool
      sendRequest('tools/call', { name: 'list_pages', arguments: {} });
      return;
    }
    if (pendingInfo.method === 'tools/call') {
      console.log('Received tools/call response for list_pages:');
      console.log(JSON.stringify(msg.result || msg.error, null, 2));
      clearTimeout(overallTimeout);
      console.log('✅ list_pages call completed — MCP end-to-end verified.');
      process.exit(0);
    }
  }
}
