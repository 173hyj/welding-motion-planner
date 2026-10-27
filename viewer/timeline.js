// Right-continuous source sampling preserves repeated phase-boundary evidence.
// Inputs are the viewer's stably time-sorted source frames.
export function selectFramePair(frames,time){
  if(!frames.length)return null;
  const stamp=index=>frames[index].time_s??frames[index].t??index;
  let lower=0,upper=frames.length;
  while(lower<upper){
    const middle=Math.floor((lower+upper)/2);
    if(stamp(middle)<=time)lower=middle+1;else upper=middle;
  }
  const index=Math.max(0,lower-1),next=Math.min(index+1,frames.length-1);
  const a=frames[index],b=frames[next],ta=stamp(index),tb=stamp(next);
  return {a,b,alpha:tb===ta?0:Math.max(0,Math.min(1,(time-ta)/(tb-ta)))};
}
