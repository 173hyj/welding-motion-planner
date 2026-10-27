export const allowedPlans=['welding_plan.json','welding_plan_three_alternative.json','welding_plan_diagnostic_v17.json','welding_plan_point7_process_release_v3.json','welding_plan_point7_strict_retreat_v4b.json','welding_plan_point7_approach_close_v5.json','welding_plan_point7_stitched_local_cycle_v3_v4_v5.json','welding_plan_point9_stitched_local_cycle_v1.json','welding_plan_point9_negative_stitched_local_cycle_v1.json','welding_plan_point7_to_point9_negative_sequence_v3.json','welding_plan_point8_negative_stitched_local_cycle_v1.json','welding_plan_point9_to_point8_negative_transfer_v1.json.gz','welding_plan_point8_to_point9_negative_transfer_v1.json.gz','welding_plan_point9_point8_roundtrip_negative_sequence_v1.json.gz'];

export function experimentWeldId(plan,sample){
 return sample?.weld_id??plan?.summary?.weld_id??plan?.status?.weld_id??plan?.frames?.find(frame=>frame.weld_id)?.weld_id??'单点';
}
export function experimentWeldLabel(plan,sample){
 const id=experimentWeldId(plan,sample),number=String(id).match(/POINT-GW-(\d+)$/)?.[1];
 return number?`Point${number}`:id;
}
export function experimentalContactLabel(sample){
 if(sample.raw_collision===true||sample.process_contact_permitted===true||sample.derived_source_face_contact_certified===true)return '有接触记录';
 if(sample.raw_collision===false||sample.strict_collision_free===true)return '电极已分离';
 return '阶段采样';
}
