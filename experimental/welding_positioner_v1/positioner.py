"""External-axis motion recovered from original PSZ JT and attachments."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from import_project import xml_root
from welding_kinematics import axis_matrix

ROOT=Path(__file__).resolve().parents[2]


def angle_grid():
    return [0]+[sign*a for a in range(5,180,5) for sign in (1,-1)]+[180]


def transform_weld(scene,weld,delta):
    product=next(a for a in scene['assets'] if a['id']==weld['product_id'])
    pose=np.eye(4);pose[:3,3]=weld['world_xyz']
    pose[:3,:3]=np.asarray(product['transform']).reshape(4,4)[:3,:3]@Rotation.from_euler('xyz',weld['projected_local_pose'][3:]).as_matrix()
    pose=delta@pose
    moved=copy.deepcopy(weld);moved['world_xyz']=pose[:3,3].tolist()
    return moved,pose


class SourcePositioner:
    def __init__(self,scene):
        self.asset=next(a for a in scene['assets'] if a['name']=='ST1-LZ-OP10L')
        raw=json.loads((ROOT/'build/positionerL_detail.json').read_text())
        text=raw['kinematicsSource'];text=text[text.index('<devices>'):text.index('</devices>')+10]
        device=ET.fromstring(text).find('device')
        joints=list(device.find('joints'));assert len(joints)==1
        joint=joints[0];assert joint.findtext('jointType')=='0'
        self.moving_link=joint.findtext('childLink')
        self.static_link=joint.findtext('parentLink')
        self.bounds=[float(joint.findtext('jointRange/'+s)) for s in ['lowLimit','highLimit']]
        self.base=np.asarray(self.asset['transform']).reshape(4,4)
        local=np.array([[float(joint.findtext(field+'/'+axis)) for axis in ['x','y','z']] for field in ['axis1','axis2']])
        self.axis_points_world=local@self.base[:3,:3].T+self.base[:3,3]
        self.axis_world=self.axis_points_world[1]-self.axis_points_world[0]
        self.axis_world/=np.linalg.norm(self.axis_world)
        link=next(n for n in raw['nodes'] if n.get('properties',{}).get('TxGmProtoKey')==self.moving_link)
        self.moving_guid=link['properties']['TxGlobalUniqueId']
        eng=xml_root((ROOT/'source/psz/EngData/engData_ver2.xml').read_bytes())
        expected='PmExtendedGuid#'+self.asset['id']+'#'+self.moving_guid
        self.attached_product_ids=[]
        product_ids={a['id']for a in scene['assets']if a['kind']=='product'}
        for obj in eng.find('eMsObjects'):
            ref=obj.find('attachment/refObject')
            if ref is not None and ref.get('IDREF')==expected:
                pid=obj.get('ID').split('#')[-1]
                if pid in product_ids:self.attached_product_ids.append(pid)
        assert set(self.attached_product_ids)==product_ids
        self.metadata=dict(source_detail='build/positionerL_detail.json',joint_key=joint.get('key'),
            moving_link=self.moving_link,static_link=self.static_link,moving_guid=self.moving_guid,
            axis_points_world_mm=self.axis_points_world.tolist(),axis_world=self.axis_world.tolist(),
            limits_rad=self.bounds,attached_product_ids=self.attached_product_ids,
            source_attachment_reference=expected,home_angle_rad=float(device.findtext('homePose/values/value')))

    def delta(self,angle_deg):
        angle=np.deg2rad(angle_deg)
        if not self.bounds[0]-1e-7<=angle<=self.bounds[1]+1e-7:raise ValueError('external axis outside source limits')
        return axis_matrix(self.axis_points_world[0],self.axis_world,angle)
