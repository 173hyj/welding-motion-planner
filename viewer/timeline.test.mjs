import test from 'node:test';
import assert from 'node:assert/strict';
import {selectFramePair} from './timeline.js';

test('equal-time seams select the new phase and its exact frame',()=>{
  const frames=[{time_s:0,phase:'open'},{time_s:1,phase:'open'},{time_s:1,phase:'retreat'},{time_s:1,phase:'retreat'},{time_s:2,phase:'retreat'}];
  const pair=selectFramePair(frames,1);
  assert.equal(pair.a,frames[3]);assert.equal(pair.b,frames[4]);assert.equal(pair.alpha,0);
});
test('interpolated geometry retains current phase until next boundary',()=>{
  const frames=[{time_s:0,phase:'open'},{time_s:1,phase:'retreat'}];
  const pair=selectFramePair(frames,.75);
  assert.equal(pair.a.phase,'open');assert.equal(pair.alpha,.75);
});
test('ends and repeated singleton states remain selectable',()=>{
  const frames=[{time_s:0},{time_s:2},{time_s:2}];
  const pair=selectFramePair(frames,2);
  assert.equal(pair.a,frames[2]);assert.equal(pair.b,frames[2]);assert.equal(pair.alpha,0);
  assert.equal(selectFramePair([],0),null);
});
