import test from 'node:test';
import assert from 'node:assert/strict';
import {gzipSync} from 'node:zlib';
let readPlanResponse;
try { ({readPlanResponse}=await import('./plan-response.js')); }
catch(error) { if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;readPlanResponse=(response)=>response.text(); }
const source='{"frames":[{"time_s":0}],"note":"焊接"}';
test('plain plan response preserves the exact audited source text',async()=>{
 assert.equal(await readPlanResponse(new Response(source),'plan.json'),source);
});
test('gzip plan response restores the exact source text for audit and playback',async()=>{
 assert.equal(await readPlanResponse(new Response(gzipSync(source)),'plan.json.gz'),source);
});
test('corrupt gzip fails rather than becoming a trajectory',async()=>{
 await assert.rejects(()=>readPlanResponse(new Response('not gzip'),'plan.json.gz'));
});
