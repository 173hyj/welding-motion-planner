import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
let options;
try{options=await import('./plan-options.js');}
catch(error){
 if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;
 const app=await fs.readFile(new URL('./app.js',import.meta.url),'utf8');
 const old=JSON.parse(app.match(/const allowedPlans=(\[[^;]+\]);/)[1].replaceAll("'",'"'));
 options={allowedPlans:old,experimentWeldId:()=> 'OP10L-SUB1-POINT-GW-7',experimentWeldLabel:()=> 'Point7'};
}
test('Point8, Point9 and continuous two-point reviewed derivatives are supported',()=>{
 for(const name of ['welding_plan_point9_stitched_local_cycle_v1.json','welding_plan_point9_negative_stitched_local_cycle_v1.json','welding_plan_point7_to_point9_negative_sequence_v3.json','welding_plan_point8_negative_stitched_local_cycle_v1.json','welding_plan_point9_to_point8_negative_transfer_v1.json.gz','welding_plan_point8_to_point9_negative_transfer_v1.json.gz','welding_plan_point9_point8_roundtrip_negative_sequence_v1.json.gz'])assert.ok(options.allowedPlans.includes(name));
});
test('original Point7 and historical plans retain support',()=>{
 for(const p of ['welding_plan.json','welding_plan_three_alternative.json','welding_plan_point7_stitched_local_cycle_v3_v4_v5.json'])assert.ok(options.allowedPlans.includes(p));
 assert.ok(!options.allowedPlans.includes('../unrelated.json'));
});
test('experimental caption uses the actual weld id instead of Point7 constant',()=>{
 const p={summary:{weld_id:'OP10L-SUB1-POINT-GW-9'}};
 assert.equal(options.experimentWeldId(p),'OP10L-SUB1-POINT-GW-9');
 assert.equal(options.experimentWeldLabel(p),'Point9');
 assert.equal(options.experimentWeldId(p,{weld_id:'OP10L-SUB1-POINT-GW-5'}),'OP10L-SUB1-POINT-GW-5');
});
test('Point7 label and frame fallback remain correct',()=>{
 assert.equal(options.experimentWeldLabel({summary:{weld_id:'OP10L-SUB1-POINT-GW-7'}}),'Point7');
 assert.equal(options.experimentWeldId({frames:[{weld_id:'OP10L-SUB1-POINT-GW-9'}]}),'OP10L-SUB1-POINT-GW-9');
});
test('experimental caption does not label a closing sample as strict entry',()=>{
 const label=options.experimentalContactLabel??(()=> '严格入口采样');
 assert.equal(label({phase:'close'}),'阶段采样');
 assert.equal(label({strict_collision_free:true}),'电极已分离');
 assert.equal(label({derived_source_face_contact_certified:true}),'有接触记录');
});
