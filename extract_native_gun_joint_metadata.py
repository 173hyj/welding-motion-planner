from pathlib import Path
import struct,zlib,lzma,re,json,hashlib,xml.etree.ElementTree as ET
root=Path('/home/ubuntu22/桌面/上航发_焊接规划_20261006');out=root/'build/raw_gun_segment_scan';out.mkdir(exist_ok=True);records=[]
for p in (root/'source/psz/EngData').glob('*.cojt/*.jt'):
 if p.parent.name not in ['C979638D-225D-4EAA-A0BD-283D3CB248FA.cojt','AC9A3D09-D9B8-4DF4-819E-28B21854297B.cojt']:continue
 b=p.read_bytes();v=b[:80].decode('latin1');major=int(re.search(r'(\d+)\.(\d+)',v)[1]);size=32 if major>9 else 28;toc=struct.unpack_from('<q' if major>9 else '<i',b,85)[0];num=struct.unpack_from('<i',b,toc)[0]
 assert b[80]==0
 for i in range(num):
  t=toc+4+i*size;off=struct.unpack_from('<q' if major>9 else '<i',b,t+16)[0];typ,length=struct.unpack_from('<ii',b,off+16);assert 24<=length and off+length<=len(b);data=b[off+24:off+length]
  if len(data)>9:
   flag,n=struct.unpack_from('<ii',data);algo=data[8]
   if flag==2 and algo==2:data=zlib.decompress(data[9:9+n-1])
   elif flag==3 and algo==3:data=lzma.decompress(data[9:9+n-1])
  for enc,shifts in [('utf-16le',[0,1]),('latin1',[0])]:
   for shift in shifts:
    txt=data[shift:].decode(enc,errors='ignore');start=txt.find('<devices>');end=txt.find('</devices>')
    if start>=0 and end>start:
     xml=txt[start:end+10];tree=ET.fromstring(xml);dest=out/f'{p.stem}_{i}_{typ}.xml';dest.write_text(xml);records.append({'source':str(p.relative_to(root)),'source_sha256':hashlib.sha256(b).hexdigest(),'segment':i,'type':typ,'xml':str(dest.relative_to(root)),'xml_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'joint_count':len(tree.findall('device/joints/joint'))})
(out/'provenance.json').write_text(json.dumps(records,indent=2,ensure_ascii=False));print('native metadata records',len(records))
