/** Apply later evidence only to the immutable motion file that was audited. */
export async function applyPlanAudit(plan, sourceText, planName, audit) {
  const entry=audit?.plans?.[planName];
  if(!entry)return {applied:false,reason:'no_audit'};
  const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(sourceText));
  const hash=Array.from(new Uint8Array(bytes),value=>value.toString(16).padStart(2,'0')).join('');
  if(entry.sha256!==hash)return {applied:false,reason:'hash_mismatch',sha256:hash};
  plan.status={...(plan.status??{}),...(entry.status??{}),audit_note:entry.note,historical_sampling_replay:entry.historical_sampling_replay??true};
  return {applied:true,sha256:hash,evidence:entry.evidence};
}
