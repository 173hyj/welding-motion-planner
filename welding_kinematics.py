"""Source JT screw axes, FANUC coupling, dependent joint polygons and crank."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def axis_matrix(point, axis, value):
    rotation = Rotation.from_rotvec(np.array(axis) / np.linalg.norm(axis) * value).as_matrix()
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = point - rotation @ point
    return result


def in_polygon(point, polygon, tolerance=1e-7):
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        aa, bb = np.array(a), np.array(b)
        d = bb - aa
        if np.dot(d, d) > 0:
            nearest = aa + np.clip(np.dot(np.array(point) - aa, d) / np.dot(d, d), 0, 1) * d
            if np.linalg.norm(nearest - point) < tolerance:
                return True
        if (a[1] > y) != (b[1] > y):
            crossing = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < crossing:
                inside = not inside
    return inside


class RobotKinematics:
    def __init__(self, scene, asset):
        self.scene = scene
        self.asset = asset
        self.joints = scene["joints"]
        self.base = np.array(asset["transform"]).reshape(4, 4)
        self.q_initial = np.array(asset["robot"]["q_initial"])
        self.bounds = []
        self.polygons = {}
        self.key_index = {j["key"]: i for i, j in enumerate(self.joints[:6])}
        for i, joint in enumerate(self.joints[:6]):
            r = ET.fromstring(joint["range"])
            if r.findtext("limitType") == "1":
                self.bounds.append((float(r.findtext("lowLimit")), float(r.findtext("highLimit"))))
            else:
                a = [float(x.text) for x in r.find("leadPoints")]
                b = [float(x.text) for x in r.find("dependPoints")]
                self.polygons[i] = (self.key_index[r.findtext("rangeJoint")], list(zip(a, b)))
                self.bounds.append((min(b), max(b)))
        self.bounds = np.array(self.bounds).T
        self.tcp_home = np.array(asset["robot"]["tcp_home"]).reshape(4, 4) if "tcp_home" in asset["robot"] else None

    def limits_valid(self, q):
        if np.any(q < self.bounds[0] - 1e-6) or np.any(q > self.bounds[1] + 1e-6):
            return False
        return all(in_polygon((q[leading], q[i]), polygon) for i, (leading, polygon) in self.polygons.items())

    def links(self, q, world=False):
        transform = np.eye(4)
        result = {"239": transform.copy()}
        for i, joint in enumerate(self.joints[:6]):
            point = np.array(joint["axis1"])
            axis = np.array(joint["axis2"]) - point
            value = q[i] + sum(q[self.key_index[k]] * f for k, f in zip(joint["leadingJoints"], joint["leadingFactors"]))
            transform = transform @ axis_matrix(point, axis, value)
            result[str(joint["childLink"])] = transform.copy()
        # Original counterbalance cylinder: endpoints fixed on links 241 and 240.
        a = np.array(self.joints[6]["axis1"])
        b = np.array(self.joints[8]["axis1"])
        pa = result["241"][:3, :3] @ a + result["241"][:3, 3]
        pb = result["240"][:3, :3] @ b + result["240"][:3, 3]
        d = pa - pb
        home = a - b
        # Crank orientation is relative to link 240. Projecting the pin span
        # in the robot base frame would fold first-axis rotation into this
        # angle a second time and incorrectly move the counterbalance parts.
        local_span = result["240"][:3, :3].T @ d
        angle = np.arctan2(home[2], home[0]) - np.arctan2(local_span[2], local_span[0])
        orientation = result["240"][:3, :3] @ Rotation.from_rotvec([0, angle, 0]).as_matrix()
        for key, p, now in (("246", a, pa), ("247", b, pb)):
            t = np.eye(4)
            t[:3, :3] = orientation
            t[:3, 3] = now - orientation @ p
            result[key] = t
        return {k: self.base @ t for k, t in result.items()} if world else result

    def tcp(self, q):
        if self.tcp_home is None:
            raise ValueError("TCP registration has not been verified")
        return self.base @ self.links(q)["245"] @ self.tcp_home

    def tool_pose_from_source_tcpf(self, q, tcpf, ref_to_natural):
        """Convert an original EngData tcpf pose after a verified reference-frame map."""
        source = np.array(tcpf, dtype=float).reshape(4, 4)
        mapping = np.array(ref_to_natural, dtype=float).reshape(4, 4)
        return self.base @ self.links(q)["245"] @ mapping @ source
