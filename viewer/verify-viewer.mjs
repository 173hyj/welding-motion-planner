import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {allowedPlans,experimentWeldId,experimentWeldLabel} from './plan-options.js';

const ROOT=path.resolve(import.meta.dirname,'..');
const planName=process.argv.find(arg=>!arg.startsWith('--')&&arg!==process.argv[0]&&arg!==process.argv[1])??'welding_plan.json';
const outArg=process.argv.find(arg=>arg.startsWith('--outdir='))?.slice('--outdir='.length);
if(!allowedPlans.includes(planName))throw new Error('Unsupported plan');
const out=outArg?path.resolve(ROOT,outArg):path.resolve(ROOT,'build/viewer_verification_'+(planName==='welding_plan.json'?'two_original_105':'three_alternative_210'));await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const sourceData=await fs.readFile(path.join(ROOT,'build/scene.json')),motionArtifactData=await fs.readFile(path.join(ROOT,'build',planName)).catch(()=>null);
const motionData=motionArtifactData&&planName.endsWith('.gz')?gunzipSync(motionArtifactData):motionArtifactData;
const report={sceneSHA256:crypto.createHash('sha256').update(sourceData).digest('hex'),
  motionArtifactSHA256:motionArtifactData?crypto.createHash('sha256').update(motionArtifactData).digest('hex'):null,
  motionSHA256:motionData?crypto.createHash('sha256').update(motionData).digest('hex'):null,
  motionStatus:motionData?(JSON.parse(motionData).status??JSON.parse(motionData).summary):null,views:[],errors:[]};
function verifyExactSourceFrame(sample,frame){
  if(Math.abs(sample.state.time-frame.time_s)>1e-10)throw new Error('Timeline quantized source timestamp');
  let maxError=0,count=0;
  for(const [gunId,gun] of Object.entries(frame.guns??{})){
    for(const [partId,source] of Object.entries(gun.parts??{})){
      const actual=sample.state.gunParts?.[`${gunId}:${partId}`];
      if(!actual||actual.length!==16)throw new Error('Missing rendered source gun part');
      for(let i=0;i<16;i++)maxError=Math.max(maxError,Math.abs(actual[i]-source[(i%4)*4+Math.floor(i/4)]));
      count++;
    }
  }
  if(maxError>1e-6)throw new Error('Rendered matrices disagree with exact source frame');
  if(count!==12)throw new Error('Expected all 12 original articulated gun parts');
  return {parts:count,maxMatrixError:maxError,seekErrorS:Math.abs(sample.state.time-frame.time_s)};
}
function diff(a,b){const pa=PNG.sync.read(a),pb=PNG.sync.read(b);let changed=0;for(let i=0;i<pa.data.length;i+=4)if(Math.abs(pa.data[i]-pb.data[i])+Math.abs(pa.data[i+1]-pb.data[i+1])+Math.abs(pa.data[i+2]-pb.data[i+2])>30)changed++;return changed/(pa.width*pa.height);}
async function canvasImage(page){return Buffer.from(await page.evaluate(()=>document.querySelector('canvas').toDataURL().split(',')[1]),'base64');}
async function restart(page){
  if(await page.locator('#restart').isVisible())await page.locator('#restart').click();
  else await page.locator('#timeline').evaluate(element=>{element.value='0';element.dispatchEvent(new Event('input',{bubbles:true}));});
}
for(const viewport of [{width:1440,height:900},{width:390,height:844}]){
  const page=await browser.newPage({viewport,deviceScaleFactor:1});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8008/viewer/?plan='+encodeURIComponent(planName),{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.viewerState?.ready,null,{timeout:180000});
  await page.waitForTimeout(1500);
  const canvas=page.locator('canvas'),before=await canvasImage(page);
  const pixels=PNG.sync.read(before);let min=255,max=0,different=0;
  for(let i=0;i<pixels.data.length;i+=4){const value=pixels.data[i];min=Math.min(min,value);max=Math.max(max,value);if(Math.abs(pixels.data[i]-232)+Math.abs(pixels.data[i+1]-237)+Math.abs(pixels.data[i+2]-233)>30)different++;}
  const state=await page.evaluate(()=>window.viewerState);
  const loadedPlan=motionData?JSON.parse(motionData):null;
  const displayedBadge=await page.locator('#plan-badge').textContent();
  const caption=await page.locator('#scene-plan-caption').textContent();
  if(['welding_plan_point9_stitched_local_cycle_v1.json','welding_plan_point9_negative_stitched_local_cycle_v1.json','welding_plan_point8_negative_stitched_local_cycle_v1.json'].includes(planName)){
    if(!displayedBadge.includes(experimentWeldLabel(loadedPlan))||!caption.includes(experimentWeldId(loadedPlan))||!caption.includes('0/29')||state.planAudit?.applied!==true)throw new Error('Local-cycle caption or exact audit binding missing');
  }
  if(planName.endsWith('.gz')&&(!displayedBadge.includes(loadedPlan.summary.sequence_label)||!caption.includes('0/29')||state.planAudit?.applied!==true))throw new Error('Compressed sequence exact audit binding/caption missing');
  if(planName==='welding_plan_point7_to_point9_negative_sequence_v3.json'&&(!displayedBadge.includes('Point7→Point9')||!caption.includes('0/29')||state.planAudit?.applied!==true))throw new Error('Two-point exact audit binding/caption missing');
  if(state.meshCount<1||different/(pixels.width*pixels.height)<.01)throw new Error('Blank or missing source geometry');
  await page.screenshot({path:path.join(out,`${viewport.width}-initial.png`)});
  await page.locator('#show-robot').uncheck();await page.waitForTimeout(200);const hidden=await canvasImage(page);await page.locator('#show-robot').check();
  const robotToggleDifference=diff(before,hidden);if(robotToggleDifference<.001)throw new Error('Robot visibility control has no pixel effect');
  if(viewport.width<640)await page.locator('#toggle-sidebar').click();
  await page.locator('.weld-row').first().click();const selected=await page.evaluate(()=>window.viewerState.selected);if(!selected)throw new Error('Weld selection failed');
  await page.locator('[data-view="top"]').click();await page.waitForTimeout(300);const top=await canvasImage(page);if(diff(before,top)<.02)throw new Error('Camera control has no effect');
  await page.locator('[data-view="perspective"]').click();
  let motionDifference=null;const phases=[];
  if(state.frames>1){
    await restart(page);await page.waitForTimeout(300);
    const motionStart=await canvasImage(page);
    const duration=await page.locator('#timeline').getAttribute('max');
    await page.locator('#speed').selectOption('1');await page.locator('#play').click();await page.waitForTimeout(Math.min(1500,Number(duration)*250));await page.locator('#play').click();
    const advancedTime=await page.evaluate(()=>window.viewerState.time);
    motionDifference=diff(motionStart,await canvasImage(page));
    if(!(advancedTime>0)||motionDifference<.0001)throw new Error('Actual joint-motion replay does not advance visible geometry');
    await restart(page);
  }
  const sourceFrames=motionData?JSON.parse(motionData).frames:[];
  const phaseSegments=[];
  for(const frame of sourceFrames){if(!frame.phase)continue;const prior=phaseSegments.at(-1);if(!prior||prior.phase!==frame.phase)phaseSegments.push({phase:frame.phase,frames:[frame]});else prior.frames.push(frame);}
  for(const [phaseIndex,phaseGroup] of phaseSegments.entries()){
    const phase=phaseGroup.phase,segment=phaseGroup.frames,first=segment[0],last=segment.at(-1);
    const seek=async time=>{await page.locator('#timeline').evaluate((element,value)=>{element.value=value;element.dispatchEvent(new Event('input',{bubbles:true}));},String(time));await page.waitForTimeout(180);return {image:await canvasImage(page),state:await page.evaluate(()=>window.viewerState)};};
    const start=await seek(first.time_s),end=await seek(last.time_s);
    const exactStart=verifyExactSourceFrame(start,first),exactEnd=verifyExactSourceFrame(end,last);
    const pixelDifference=diff(start.image,end.image);
    const partDifference=Object.keys(start.state.gunParts??{}).filter(id=>JSON.stringify(start.state.gunParts[id])!==JSON.stringify(end.state.gunParts?.[id])).length;
    if(!(last.time_s>first.time_s)||!partDifference||pixelDifference<.000001)throw new Error('Phase has no true articulated motion: '+phase);
    const middle=Math.min(segment.length-2,Math.floor(segment.length/2)), adjacentA=segment[middle],adjacentB=segment[middle+1];
    const adjacentStart=await seek(adjacentA.time_s),adjacentEnd=await seek(adjacentB.time_s);
    const exactAdjacentStart=verifyExactSourceFrame(adjacentStart,adjacentA),exactAdjacentEnd=verifyExactSourceFrame(adjacentEnd,adjacentB);
    const adjacentPixelDifference=diff(adjacentStart.image,adjacentEnd.image);
    const adjacentPartDifference=Object.keys(adjacentStart.state.gunParts??{}).filter(id=>JSON.stringify(adjacentStart.state.gunParts[id])!==JSON.stringify(adjacentEnd.state.gunParts?.[id])).length;
    if(!(adjacentB.time_s>adjacentA.time_s)||!adjacentPartDifference||!adjacentPixelDifference)throw new Error('Adjacent source frames do not render motion: '+phase);
    phases.push({index:phaseIndex+1,phase,weldId:first.weld_id,firstTime:first.time_s,lastTime:last.time_s,samples:segment.length,exactStart,exactEnd,exactAdjacentStart,exactAdjacentEnd,pixelDifference,movingGunParts:partDifference,adjacentFrameDeltaS:adjacentB.time_s-adjacentA.time_s,adjacentPixelDifference,adjacentMovingGunParts:adjacentPartDifference});
    await page.screenshot({path:path.join(out,`${viewport.width}-${String(phaseIndex+1).padStart(2,'0')}-${phase}.png`)});
  }
  await restart(page);
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth||document.documentElement.scrollHeight>innerHeight);
  if(overflow)throw new Error('Viewport overflow');
  await page.screenshot({path:path.join(out,`${viewport.width}-selected.png`)});
  report.views.push({viewport,state,displayedBadge,caption,pixelRange:[min,max],nonBackgroundFraction:different/(pixels.width*pixels.height),robotToggleDifference,motionDifference,phases,overflow,errors});
  report.errors.push(...errors);await page.close();
}
await browser.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.errors.length)process.exitCode=1;
