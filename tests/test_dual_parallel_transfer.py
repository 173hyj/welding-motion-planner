import numpy as np
from experimental.dual_arm_parallel_v1.transfer import make_transfer


class Bridge:
    robot_ids=['a','b']
    scale=np.r_[np.ones(13),100.,100.]
    low=np.r_[np.full(13,-2.),-185.,-185.]
    high=np.r_[np.full(13,2.),0.,0.]
    resolutions=np.r_[np.full(12,np.deg2rad(.05)),np.deg2rad(.25),.5,.5]
    checks=0
    def __init__(self, obstacle=False): self.obstacle=obstacle; self.seen=[]
    def valid_scaled(self,s):
        s=np.asarray(s)*self.scale;self.seen.append(s.copy()); self.checks+=1
        return not (self.obstacle and -.15<s[0]<.15 and abs(s[1])<.2)
    def edge_scaled(self,a,b):
        n=max(1,int(np.ceil(np.max(abs(np.asarray(b)-a)/(np.r_[np.full(12,np.deg2rad(1.)),np.deg2rad(1.),2.,2.]/self.scale)))))
        return all(self.valid_scaled(np.asarray(a)+(np.asarray(b)-a)*u) for u in np.linspace(0,1,n+1))


def endpoints():
    a=np.array([-.5,0,0,0,0,-.5,-100.]);b=a.copy();b[0]=.5
    park=np.array([1.,0,0,0,0,.5,0.])
    return a,b,park


def test_reuses_matching_reverse_edge_with_other_arm_frozen_and_fine_certifies():
    a,b,park=endpoints();mid=(a+b)/2;mid[1]=.4
    bridge=Bridge(True)
    path,stats=make_transfer(None,bridge,0,30,a,b,[np.array([b,mid,a])],park7=park)
    assert stats['method']=='existing_edge' and stats['fine_certified']
    assert np.array_equal(path[[0,-1]],[a,b])
    assert all(np.array_equal(s[6:12],park[:6]) and s[14]==park[6] and s[12]==np.deg2rad(30) for s in bridge.seen)
    assert len(bridge.seen)>100


def test_direct_and_rejected_rrt_fail_closed():
    a,b,park=endpoints()
    path,stats=make_transfer(None,Bridge(),0,0,a,b,park7=park)
    assert np.array_equal(path,[a,b]) and stats['method']=='direct'
    path,stats=make_transfer(None,Bridge(True),0,0,a,b,park7=park,max_iterations=0,enable_gateways=False)
    assert path is None and not stats['fine_certified']


def test_bounded_rrt_reproducibly_avoids_obstacle_without_endpoint_changes():
    a,b,park=endpoints()
    path,stats=make_transfer(None,Bridge(True),0,0,a,b,park7=park,seed=4,max_iterations=300,enable_gateways=False)
    assert path is not None and stats['method']=='rrt' and stats['fine_certified']
    assert np.array_equal(path[[0,-1]],[a,b])


def test_closing_gun_before_motion_avoids_open_gun_obstacle():
    class GunObstacle(Bridge):
        def valid_scaled(self,s):
            state=np.asarray(s)*self.scale;self.checks+=1;self.seen.append(state.copy())
            return not (-.15<state[0]<.15 and state[13]<-1.)
    a,b,park=endpoints();bridge=GunObstacle()
    path,stats=make_transfer(None,bridge,0,0,a,b,park7=park,max_iterations=0)
    assert stats['method']=='close_move_open' and stats['fine_certified']
    assert np.array_equal(path[[0,-1]],[a,b])
    assert path[1,6]==path[2,6]==0.


def test_folding_at_endpoint_before_base_rotation():
    a,b,park=endpoints()
    path,stats=make_transfer(None,Bridge(True),0,0,a,b,park7=park,max_iterations=0)
    assert stats['fine_certified'] and stats['method']=='endpoint_fold_rotate'
    assert path[2,0]==a[0] and path[3,0]==b[0]
