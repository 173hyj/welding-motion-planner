import test from 'node:test';
import assert from 'node:assert/strict';
import {applyPlanAudit} from './plan-audit.js';
const text='{"status":{"local_sequence_collision_verified":true}}';
const hash=Buffer.from(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))).toString('hex');
test('Exact file hash binds superseding diagnostic audit',async()=>{
 const plan=JSON.parse(text);const audit={plans:{'old.json':{sha256:hash,status:{diagnostic_only:true,local_sequence_collision_verified:false},note:'密采样复核失败'}}};
 const result=await applyPlanAudit(plan,text,'old.json',audit);
 assert.equal(result.applied,true);assert.equal(plan.status.local_sequence_collision_verified,false);assert.equal(plan.status.diagnostic_only,true);assert.equal(plan.status.audit_note,'密采样复核失败');
});
test('Mismatched motion cannot receive unrelated audit',async()=>{
 const plan=JSON.parse(text);const result=await applyPlanAudit(plan,text,'old.json',{plans:{'old.json':{sha256:'other',status:{diagnostic_only:true}}}});
 assert.equal(result.applied,false);assert.equal(plan.status.local_sequence_collision_verified,true);
});
