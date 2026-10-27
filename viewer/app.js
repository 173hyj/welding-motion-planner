import * as THREE from 'three';
import {applyPlanAudit} from './plan-audit.js';
import {readPlanResponse} from './plan-response.js';
import {selectFramePair} from './timeline.js';
import {allowedPlans,experimentWeldId,experimentWeldLabel,experimentalContactLabel} from './plan-options.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {createIcons, icons} from 'lucide';

const $ = id => document.getElementById(id);
createIcons({icons});
const host = $('canvas-host');
const renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0xe8ede9);
host.append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42,1,.01,200);
camera.up.set(0,1,0);
camera.position.set(7,-9,6);
const controls = new OrbitControls(camera,renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = .09;
scene.add(new THREE.HemisphereLight(0xffffff,0x7c8f79,2.8));
const sun = new THREE.DirectionalLight(0xffffff,3.2);
sun.position.set(5,-6,10); scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff,1.4);
fill.position.set(-5,4,4); scene.add(fill);
const world = new THREE.Group(); world.scale.setScalar(.001); scene.add(world);
const modelRoot = new THREE.Group();world.add(modelRoot);
const robotRoot = new THREE.Group();world.add(robotRoot);
const markerRoot = new THREE.Group();world.add(markerRoot);
const trajectoryRoot = new THREE.Group();world.add(trajectoryRoot);
const grid = new THREE.GridHelper(20,40,0xb6c5ba,0xd5ddd7);grid.position.y=-.003;scene.add(grid);
const loader = new GLTFLoader();
const meshCache = new Map();
const assetObjects = [], linkObjects = new Map(), markers = new Map();
const displayUrls = new Map(), gunObjects = new Map();
const gunPartObjects = new Map();
let sceneData, planData, frames=[], currentTime=0, duration=0, playing=false;
let selectedId=null, welds=[], bounds=new THREE.Box3(), center=new THREE.Vector3(), extent=5, activePhase='';
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
window.viewerState={ready:false, meshCount:0, frames:0, selected:null, playing:false};

function matrix(values){return new THREE.Matrix4().set(...values);}
function setTransform(object,values){if(values?.length===16){object.matrix.copy(matrix(values));object.matrixAutoUpdate=false;}}
function urlFor(url){return new URL(url, new URL('../build/scene.json',location.href)).href;}
function clockLabel(seconds){return `${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(Math.floor(seconds%60)).padStart(2,'0')}`;}
const phaseNames={open:'开枪',opening:'开枪',retreat:'退刀',transfer:'转移',approach:'进刀',close:'合枪',experimental_release:'工艺退离（实验）',point9_derived_release:'工艺退离（实验）',controlled_approach:'受控进刀（实验）',return:'返回分离点'};
function status(text,error=false){$('load-status').innerHTML=`<span class="status-dot"></span>${text}`;$('load-status').classList.toggle('error',error);}
function text(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

async function loadMesh(url,parent,transform,name,kind){
  url=displayUrls.get(url)??url;
  if(!meshCache.has(url))meshCache.set(url,loader.loadAsync(urlFor(url)));
  const gltf=await meshCache.get(url),model=gltf.scene.clone(true);
  const wrapper=new THREE.Group(); wrapper.name=name;wrapper.add(model);setTransform(wrapper,transform);parent.add(wrapper);
  model.traverse(object=>{if(object.isMesh){window.viewerState.meshCount++;object.material.side=THREE.DoubleSide;object.castShadow=false;object.receiveShadow=false;}});
  assetObjects.push({object:wrapper,kind});return wrapper;
}

function fit(view='perspective'){
  modelRoot.updateWorldMatrix(true,true);robotRoot.updateWorldMatrix(true,true);
  bounds.makeEmpty();bounds.expandByObject(robotRoot);
  for(const asset of assetObjects.filter(a=>a.kind==='product'))bounds.expandByObject(asset.object);
  if(bounds.isEmpty())bounds.expandByObject(modelRoot);
  if(bounds.isEmpty())return;
  bounds.getCenter(center);extent=Math.max(.5,bounds.getSize(new THREE.Vector3()).length());
  const directions={perspective:new THREE.Vector3(1,.72,1),top:new THREE.Vector3(0,1,.0001),front:new THREE.Vector3(0,.18,1)};
  const distance=extent/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*Math.min(1,camera.aspect))*1.04;
  camera.position.copy(center).addScaledVector(directions[view].normalize(),distance);
  camera.near=Math.max(.005,extent/1000);camera.far=Math.max(100,extent*30);camera.updateProjectionMatrix();
  controls.target.copy(center);controls.maxDistance=extent*10;controls.update();
  grid.scale.setScalar(Math.max(1,extent/10));grid.position.set(center.x,bounds.min.y-.008,center.z);
}

function buildWelds(){
  welds=sceneData.welds??[];
  $('weld-count').textContent=welds.length;
  const groups=[...new Set(welds.map(w=>w.group??'未分组'))];
  $('group').innerHTML='<option value="">全部分组</option>'+groups.map(g=>`<option>${text(g)}</option>`).join('');
  const geometry=new THREE.SphereGeometry(9,12,8);
  for(const weld of welds){
    const xyz=weld.world_xyz??weld.position;
    if(!xyz?.length)continue;
    const marker=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0x8b9690,depthTest:false,transparent:true,opacity:.92}));
    marker.position.fromArray(xyz);marker.renderOrder=20;marker.userData.weldId=String(weld.id);markerRoot.add(marker);markers.set(String(weld.id),marker);
  }
  renderWeldList();
}

function renderWeldList(){
  const query=$('search').value.toLocaleLowerCase(),group=$('group').value;
  const filtered=welds.filter(w=>(!group||(w.group??'未分组')===group)&&String(w.name??w.id).toLocaleLowerCase().includes(query));
  $('weld-list').innerHTML=filtered.length?filtered.map((w,index)=>`<button class="weld-row${String(w.id)===selectedId?' selected':''}" data-weld-id="${text(w.id)}"><span class="weld-index">${String(w.sequence??index+1).padStart(2,'0')}</span><span><span class="weld-name">${text(w.name??w.id)}</span><span class="weld-group">${text(w.group??'未分组')}</span></span><i data-lucide="crosshair"></i></button>`).join(''):'<div class="empty-state">无匹配焊缝</div>';
  createIcons({icons});
  $('weld-list').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>selectWeld(button.dataset.weldId)));
}

function selectWeld(id){
  selectedId=String(id);const weld=welds.find(w=>String(w.id)===selectedId);if(!weld)return;
  for(const [key,marker] of markers){marker.material.color.setHex(key===selectedId?0x1a597d:marker.userData.statusColor??0x8b9690);marker.scale.setScalar(key===selectedId?1.7:1);}
  $('selection-info').hidden=false;$('selected-name').textContent=weld.name??weld.id;
  $('selected-coordinate').textContent=(weld.world_xyz??weld.position??[]).map(v=>Number(v).toFixed(1)).join(' · ');
  $('selected-sequence').textContent=`${weld.group??'—'} / ${weld.sequence??'—'}`;
  window.viewerState.selected=selectedId;renderWeldList();
  if(innerWidth<=640)$('sidebar').classList.remove('open');
  const frame=frames.find(f=>String(f.weld_id??f.operation_id)===selectedId);
  if(frame){currentTime=frame.time_s??frame.t??0;applyFrame(currentTime);}
}

function linkTransforms(frame){return frame.links??frame.link_transforms??frame.transforms??{};}
function applyFrame(time){
  if(!frames.length)return;
  const {a,b,alpha}=selectFramePair(frames,time);
  const tb=b.time_s??b.t??0;
  const transformsA=linkTransforms(a),transformsB=linkTransforms(b);
  for(const [id,object] of linkObjects){
    const [robotId,linkId]=id.split(':');
    const robotA=a.robots?.[robotId],robotB=b.robots?.[robotId];
    const ma=robotA?.links?.[linkId]??transformsA[id],mb=robotB?.links?.[linkId]??transformsB[id]??ma;
    if(!ma||!mb)continue;
    const p1=new THREE.Vector3(),p2=new THREE.Vector3(),q1=new THREE.Quaternion(),q2=new THREE.Quaternion(),s1=new THREE.Vector3(),s2=new THREE.Vector3();
    matrix(ma).decompose(p1,q1,s1);matrix(mb).decompose(p2,q2,s2);object.matrix.compose(p1.lerp(p2,alpha),q1.slerp(q2,alpha),s1.lerp(s2,alpha));object.matrixWorldNeedsUpdate=true;
  }
  for(const [id,object] of gunObjects){
    const ga=a.guns?.[id],gb=b.guns?.[id];
    const ma=ga?.transform??ga,mb=gb?.transform??gb??ma;if(!Array.isArray(ma)||!Array.isArray(mb))continue;
    const p1=new THREE.Vector3(),p2=new THREE.Vector3(),q1=new THREE.Quaternion(),q2=new THREE.Quaternion(),s1=new THREE.Vector3(),s2=new THREE.Vector3();
    matrix(ma).decompose(p1,q1,s1);matrix(mb).decompose(p2,q2,s2);object.matrix.compose(p1.lerp(p2,alpha),q1.slerp(q2,alpha),s1.lerp(s2,alpha));object.matrixWorldNeedsUpdate=true;
  }
  for(const [id,object] of gunPartObjects){
    const [gunId,partId]=id.split(':');const ma=a.guns?.[gunId]?.parts?.[partId],mb=b.guns?.[gunId]?.parts?.[partId]??ma;if(!ma||!mb)continue;
    const p1=new THREE.Vector3(),p2=new THREE.Vector3(),q1=new THREE.Quaternion(),q2=new THREE.Quaternion(),s1=new THREE.Vector3(),s2=new THREE.Vector3();
    matrix(ma).decompose(p1,q1,s1);matrix(mb).decompose(p2,q2,s2);object.matrix.compose(p1.lerp(p2,alpha),q1.slerp(q2,alpha),s1.lerp(s2,alpha));object.matrixWorldNeedsUpdate=true;
  }
  activePhase=a.phase??'';
  if(planData?.status?.experimental_process_release||planData?.status?.experimental_controlled_approach_close||planData?.status?.experimental_stitched_local_cycle||planData?.status?.experimental_stitched_local_sequence){
    const sample=time>=tb?b:a;
    const contact=experimentalContactLabel(sample);
    const permission=sample.process_contact_permitted?' · 工艺接触许可':sample.strict_weld_contract_verified?' · 名义焊点接触合同':'';
    $('scene-plan-caption').textContent=`${experimentWeldId(planData,sample)} · ${phaseNames[sample.phase]??sample.phase} · ${contact}${permission} · 严格规划 0/29`;
    window.viewerState.processSample={time:sample.time_s,rawContact:sample.raw_collision,permitted:sample.process_contact_permitted,state:sample.fsm_state,travelMM:sample.travel_mm};
  }
  $('timeline').value=time;$('time-label').textContent=`${clockLabel(time)} / ${clockLabel(duration)}${activePhase?` · ${phaseNames[activePhase]??activePhase}`:''}`;window.viewerState.time=time;window.viewerState.phase=activePhase;window.viewerState.gunPartCount=gunPartObjects.size;window.viewerState.gunParts=Object.fromEntries([...gunPartObjects].map(([id,o])=>[id,o.matrix.elements.slice()]));
}

function setPlaying(value){
  playing=value&&frames.length>1;window.viewerState.playing=playing;
  $('play').innerHTML=`<i data-lucide="${playing?'pause':'play'}"></i>`;$('play').title=playing?'暂停轨迹':'播放轨迹';$('play').setAttribute('aria-label',$('play').title);createIcons({icons});
}

async function loadPlan(){
  const requestedPlan=new URLSearchParams(location.search).get('plan');
  if(requestedPlan&&!allowedPlans.includes(requestedPlan))throw new Error('未识别的计划文件');
  const selectedPlan=requestedPlan??'welding_plan.json';
  const candidates=[selectedPlan];
  for(const url of candidates){
    try{const response=await fetch(urlFor(url));if(response.ok){const sourceText=await readPlanResponse(response,selectedPlan);planData=JSON.parse(sourceText);try{const auditResponse=await fetch(urlFor('eight_hour_v17/audit/plan_overrides.json'));if(auditResponse.ok)window.viewerState.planAudit=await applyPlanAudit(planData,sourceText,selectedPlan,await auditResponse.json());}catch{}break;}}catch{}
  }
  frames=planData?.motion?.frames??planData?.frames??sceneData.motion?.frames??[];
  frames.sort((a,b)=>(a.time_s??a.t??0)-(b.time_s??b.t??0));
  duration=frames.length?(frames.at(-1).time_s??frames.at(-1).t??frames.length-1):0;
  const enabled=frames.length>1&&linkObjects.size>0;
  ['play','restart','timeline'].forEach(id=>$(id).disabled=!enabled);$('timeline').max=duration||1;
  window.viewerState.frames=frames.length;
  let collisionData;
  const collisionURLs=[planData?.collision_plan,'collision_plan_source_normal_roll_optimized.json','collision_plan.json'].filter(Boolean);
  for(const url of collisionURLs){try{const response=await fetch(urlFor(url));if(response.ok){collisionData=await response.json();break;}}catch{}}
  const summary={...(sceneData.planning??{}),...(planData??{}),...(planData?.summary??{}),...(planData?.status??{})};
  const verified=(summary.collision_verified===true||summary.safety_verified===true)&&summary.tcp_verified!==false&&!summary.diagnostic_only;
  const planned=enabled;
  const localSequence=planData?.status?.local_sequence_collision_verified===true||planData?.status?.transit_collision_verified===true;
  const historical=summary.historical_sampling_replay===true;
  const processExperiment=summary.experimental_process_release===true;
  const controlledApproachExperiment=summary.experimental_controlled_approach_close===true;
  const stitchedCycle=summary.experimental_stitched_local_cycle===true;
  const stitchedSequence=summary.experimental_stitched_local_sequence===true;
  const strictRetreatExperiment=summary.strict_separated_retreat_return_samples_passed===true;
  const correctedDiagnostic=planData?.schema==='corrected_source_diagnostic_motion_v17'&&summary.diagnostic_only===true;
  $('plan-badge').textContent=stitchedSequence?`${summary.sequence_label??'Point7→Point9'}负J6连续派生序列`:stitchedCycle?`${experimentWeldLabel(planData)}局部完整循环`:strictRetreatExperiment?'无接触退刀实验':controlledApproachExperiment?'受控进刀/合枪实验':processExperiment?'单点工艺实验':correctedDiagnostic?'修正FK诊断运动':historical?'历史采样回放':verified?'已验证':localSequence?'局部序列':'未规划';$('plan-badge').className='badge '+(verified?'good':'pending');
  const endpointWelds=new Set((collisionData?.candidates??[]).filter(c=>c.collision_free_environment).map(c=>c.weld_id));
  const localWelds=new Set(planData?.selected_weld_ids??(localSequence?frames.map(f=>f.weld_id).filter(Boolean):[]));
  const alternativePark=summary.r2_alternative_park_verified===true;
  window.viewerState.processExperiment=processExperiment||controlledApproachExperiment||stitchedCycle||stitchedSequence;window.viewerState.stitchedSequence=stitchedSequence;window.viewerState.controlledApproachExperiment=controlledApproachExperiment;window.viewerState.stitchedCycle=stitchedCycle;window.viewerState.plan=selectedPlan;window.viewerState.localWeldCount=localWelds.size;window.viewerState.alternativePark=alternativePark;
  $('scene-plan-caption').textContent=correctedDiagnostic?'修正平衡缸FK · 源关节限速 · 0/29 任务通过 · 诊断运动':historical?`${localWelds.size}/${welds.length} 焊点历史运动回放 · 密采样复核失败 · 未认证安全路线`:localSequence?`${localWelds.size}/${welds.length} 焊点局部序列 · ${alternativePark?'R2 替代停放 · 源→替代未验证':'源起点不安全'} · 非全工位验证`:summary.diagnostic_only?'源位姿诊断 · 未生成焊接路线':'';
  $('plan-details').innerHTML=`<div><dt>轨迹</dt><dd>${planned?`${frames.length} 帧 · ${duration.toFixed(1)} s`:'无关节轨迹'}</dd></div><div><dt>TCP 注册</dt><dd>${summary.tcp_verified?'已核验':'待核验'}</dd></div><div><dt>连续路线</dt><dd>${verified?'已通过':summary.source_initial_unsafe?'起点存在碰撞':'未验证'}</dd></div><div><dt>局部序列</dt><dd>${localWelds.size||0} / ${welds.length}</dd></div>${collisionData?`<div><dt>无环境碰撞端点</dt><dd>${endpointWelds.size} / ${welds.length}</dd></div>`:''}`;
  const note=(processExperiment||controlledApproachExperiment||stitchedCycle||stitchedSequence||strictRetreatExperiment)?summary.note:correctedDiagnostic?summary.note:historical?summary.audit_note:summary.diagnostic_only?'仅回放源关节位姿。已找到部分无环境碰撞端点，但源起点存在自碰撞，尚无通过验证的连续焊接路线。':localSequence?`局部序列 ${localWelds.size} 个焊点已通过阶段采样；${alternativePark?'R2 采用替代停放；源到替代起点未验证':'源起点不安全'}，非全工位验证。`:summary.note??summary.reason??summary.limitation??summary.limitations??'';$('plan-note').textContent=Array.isArray(note)?note.join('；'):note;
  const statuses=planData?.welds??planData?.results??planData?.weld_results??[];
  for(const item of statuses){const marker=markers.get(String(item.id??item.weld_id));if(!marker)continue;const accepted=item.success===true||item.reachable===true||item.status==='accepted';const rejected=item.success===false||item.reachable===false||item.status==='rejected';const color=accepted?(verified?0x338761:0xb68835):rejected?0xbf554b:0x8b9690;marker.userData.statusColor=color;marker.material.color.setHex(color);}
  if(collisionData){for(const weld of welds){const marker=markers.get(String(weld.id));if(!marker)continue;const candidates=collisionData.candidates.filter(c=>String(c.weld_id)===String(weld.id));const color=!historical&&!correctedDiagnostic&&!processExperiment&&!controlledApproachExperiment&&localWelds.has(String(weld.id))&&endpointWelds.has(String(weld.id))?0x338761:endpointWelds.has(String(weld.id))?0xb68835:candidates.length?0xbf554b:0x8b9690;marker.userData.statusColor=color;marker.material.color.setHex(color);}}
  const points=frames.map(f=>f.tcp_xyz??f.tcp_position??Object.values(f.robots??{})[0]?.tcp_xyz).filter(p=>p?.length===3);
  if(points.length>1){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p))),new THREE.LineBasicMaterial({color:0x308665,transparent:true,opacity:.75}));trajectoryRoot.add(line);}
  applyFrame(0);
}

async function loadScene(){
  try{
    const response=await fetch('../build/scene.json');if(!response.ok)throw new Error('scene.json 尚未生成');sceneData=await response.json();
    try{const lod=await fetch('../build/display_meshes/manifest.json');if(lod.ok){const data=await lod.json();for(const mesh of data.meshes)displayUrls.set(mesh.source.replace(/^build\//,''),mesh.display.replace(/^build\//,''));}}catch{}
    $('project-name').textContent=sceneData.name??sceneData.project_name??'AP31-LZ · OP10';
    const jobs=[];
    for(const asset of sceneData.assets??[]){
      const kind=asset.kind??'part',robot=asset.robot;
      const links=robot?.link_models??robot?.link_urls??asset.link_urls;
      const gunLinks=kind==='gun'?(asset.gun?.link_models??asset.part_models??asset.gun?.parts):null;
      if(gunLinks){
        for(const [id,part] of Object.entries(gunLinks)){
          const local=part.current_transform??part.transform;
          const initial=part.world_transform??(local&&asset.transform?matrix(asset.transform).multiply(matrix(local)).transpose().toArray():asset.transform);
          jobs.push(loadMesh(part.url,modelRoot,initial,`${asset.name}:${id}`,'gun').then(object=>gunPartObjects.set(`${asset.id}:${id}`,object)));
        }
      }else if(links){
        const entries=Array.isArray(links)?links.map(l=>[String(l.id),l]):Object.entries(links);
        for(const [id,link] of entries){
          const url=typeof link==='string'?link:link.url;
          const local=link.current_transform??robot?.current_link_transforms?.[id]??asset.current_link_transforms?.[id];
          const rowmajor=local&&asset.transform?matrix(asset.transform).multiply(matrix(local)).transpose().toArray():local;
          jobs.push(loadMesh(url,robotRoot,rowmajor,`${asset.name}:${id}`,'robot').then(object=>linkObjects.set(`${asset.id}:${id}`,object)));
        }
      }else if(asset.url){jobs.push(loadMesh(asset.url,kind==='robot'?robotRoot:modelRoot,asset.transform,asset.name,kind).then(object=>{if(kind==='gun')gunObjects.set(String(asset.id),object);if(kind==='layout')object.visible=false;}));}
    }
    const outcomes=await Promise.allSettled(jobs);const failures=outcomes.filter(r=>r.status==='rejected');
    buildWelds();fit();await loadPlan();
    $('geometry-label').textContent=`源 JT${displayUrls.size?' · 检查 LOD':''} · ${window.viewerState.meshCount} 网格`;
    if(failures.length){status(`${failures.length} 个资源失败`,true);$('scene-message').textContent=failures.map(f=>f.reason.message).join('；');console.error(failures);}else{$('scene-message').hidden=true;status('工程已载入');}
    window.viewerState.ready=true;
  }catch(error){status('工程未就绪',true);$('scene-message').textContent=error.message;$('plan-badge').textContent='等待工程';console.error(error);}
}

$('search').addEventListener('input',renderWeldList);$('group').addEventListener('change',renderWeldList);
$('toggle-sidebar').addEventListener('click',()=>$('sidebar').classList.toggle('open'));
$('reload').addEventListener('click',()=>location.reload());$('fit').addEventListener('click',()=>fit());
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>fit(button.dataset.view)));
$('show-robot').addEventListener('change',event=>robotRoot.visible=event.target.checked);
$('show-fixture').addEventListener('change',event=>assetObjects.filter(a=>a.kind==='fixture').forEach(a=>a.object.visible=event.target.checked));
$('show-layout').addEventListener('change',event=>assetObjects.filter(a=>a.kind==='layout').forEach(a=>a.object.visible=event.target.checked));
$('show-welds').addEventListener('change',event=>{markerRoot.visible=event.target.checked;trajectoryRoot.visible=event.target.checked;});
$('play').addEventListener('click',()=>{if(currentTime>=duration)currentTime=0;setPlaying(!playing);});
$('restart').addEventListener('click',()=>{currentTime=0;setPlaying(false);applyFrame(0);});
$('timeline').addEventListener('input',event=>{currentTime=Number(event.target.value);setPlaying(false);applyFrame(currentTime);});
let pointerDown;
renderer.domElement.addEventListener('pointerdown',event=>pointerDown=[event.clientX,event.clientY]);
renderer.domElement.addEventListener('pointerup',event=>{
  if(!pointerDown||Math.hypot(event.clientX-pointerDown[0],event.clientY-pointerDown[1])>5)return;
  const rect=renderer.domElement.getBoundingClientRect();pointer.set(2*(event.clientX-rect.left)/rect.width-1,1-2*(event.clientY-rect.top)/rect.height);raycaster.setFromCamera(pointer,camera);
  const hit=raycaster.intersectObjects([...markers.values()])[0];if(hit)selectWeld(hit.object.userData.weldId);
});
new ResizeObserver(()=>{const {width,height}=host.getBoundingClientRect();renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();}).observe(host);
let last=performance.now(),dirty=true,lastRendered=0;
controls.addEventListener('change',()=>dirty=true);
document.addEventListener('click',()=>dirty=true);
document.addEventListener('input',()=>dirty=true);
document.addEventListener('change',()=>dirty=true);
new ResizeObserver(()=>dirty=true).observe(host);
function animate(now){requestAnimationFrame(animate);const delta=(now-last)/1000;last=now;if(playing){currentTime=Math.min(duration,currentTime+delta*Number($('speed').value));applyFrame(currentTime);dirty=true;if(currentTime>=duration)setPlaying(false);}controls.update();if(dirty&&now-lastRendered>33){renderer.render(scene,camera);dirty=false;lastRendered=now;}}
requestAnimationFrame(animate);loadScene();
