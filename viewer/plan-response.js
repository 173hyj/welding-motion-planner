/** Preserve exact UTF-8 motion text for the existing SHA-bound audit. */
export async function readPlanResponse(response, planName) {
 if(planName.endsWith('.gz'))return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).text();
 return response.text();
}
