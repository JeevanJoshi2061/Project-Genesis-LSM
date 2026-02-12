"""
embodiment.py - jarvis's virtual 3d body
pybullet physics: gravity, collisions, pain from impacts.
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# ---- pybullet ----
try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
    print("pybullet available")
except ImportError:
    HAS_PYBULLET = False
    print("no pybullet - pip install pybullet for 3d body")


# ---- state ----

@dataclass
class BodyState:
    """physical body state"""
    position: Tuple[float, float, float] = (0, 0, 0.5)
    orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)  # quat
    velocity: Tuple[float, float, float] = (0, 0, 0)
    angular_velocity: Tuple[float, float, float] = (0, 0, 0)
    is_grounded: bool = True
    is_colliding: bool = False
    collision_force: float = 0.0

@dataclass
class PhysicsEvent:
    """collision, fall, impact"""
    event_type: str
    force: float
    location: Tuple[float, float, float]
    timestamp: float = field(default_factory=time.time)


# ---- virtual body ----

class VirtualBody:
    """
    robot in a 3d world. falls, feels pain, learns physics like a toddler.
    """

    def __init__(self, body_type="sphere", mass=1.0,
                 on_collision=None, on_fall=None, gui=False):
        self.body_type = body_type
        self.mass = mass
        self.on_collision = on_collision
        self.on_fall = on_fall
        self.gui = gui

        self.physics_client = None
        self.body_id = None
        self.floor_id = None
        self.state = BodyState()
        self.last_height = 0.5
        self.is_falling = False
        self.events = []
        self.physics_pain = 0.0

        print(f"-- body init ({body_type}, {mass}kg, gui:{'Y' if gui else 'N'}) --")

    def spawn(self):
        """spawn body in physics world"""
        if not HAS_PYBULLET:
            print("no pybullet!")
            return False
        try:
            self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1/240)
            self.floor_id = p.loadURDF("plane.urdf")

            if self.body_type == "sphere":
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 0.5, 1, 1])
            else:  # box
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2]*3)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2]*3, rgbaColor=[0, 0.5, 1, 1])

            self.body_id = p.createMultiBody(
                baseMass=self.mass, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis, basePosition=[0, 0, 0.5])
            print("body spawned")
            return True
        except Exception as e:
            print(f"spawn failed: {e}")
            return False

    def step(self):
        """tick physics, detect falls & collisions"""
        if self.physics_client is None:
            return self.state

        p.stepSimulation()
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        vel, ang = p.getBaseVelocity(self.body_id)

        self.state.position = pos
        self.state.orientation = orn
        self.state.velocity = vel
        self.state.angular_velocity = ang
        self.state.is_grounded = pos[2] < 0.25

        # falling?
        if not self.state.is_grounded and pos[2] < self.last_height:
            if not self.is_falling:
                self.is_falling = True
                ev = PhysicsEvent("fall", 0, pos)
                self.events.append(ev)
                if self.on_fall:
                    self.on_fall(ev)
                print(f"FALLING! h={pos[2]:.2f}m")
        else:
            self.is_falling = False
        self.last_height = pos[2]

        # collisions
        contacts = p.getContactPoints(self.body_id)
        if contacts:
            total = sum(c[9] for c in contacts)
            self.state.is_colliding = True
            self.state.collision_force = total
            if total > 10:
                self.physics_pain = min(100, total / 10)
                ev = PhysicsEvent("collision", total, pos)
                self.events.append(ev)
                if self.on_collision:
                    self.on_collision(ev)
                print(f"COLLISION! {total:.1f}N pain:{self.physics_pain:.1f}")
        else:
            self.state.is_colliding = False
            self.state.collision_force = 0
            self.physics_pain = max(0, self.physics_pain - 1)

        return self.state

    def apply_force(self, force):
        if self.body_id is not None:
            p.applyExternalForce(self.body_id, -1, force,
                self.state.position, p.WORLD_FRAME)

    def move(self, direction, speed=5.0):
        forces = {
            "forward": (speed, 0, 0), "backward": (-speed, 0, 0),
            "left": (0, speed, 0), "right": (0, -speed, 0),
            "up": (0, 0, speed * 2),  # jump
        }
        if direction in forces:
            self.apply_force(forces[direction])

    def get_pain(self):
        return self.physics_pain

    def destroy(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            print("body destroyed")


# ---- physics sense (body -> neural signals for LSM) ----

class PhysicsSense:
    """convert body state to 32-dim sensory vector for the reservoir"""

    def __init__(self, body):
        self.body = body

    def get_sensory_vector(self):
        st = self.body.state

        pos = np.array(st.position)
        vel = np.array(st.velocity)
        spd = np.linalg.norm(vel)

        # balance from quaternion
        q = st.orientation
        up = 2 * (q[0]*q[2] + q[1]*q[3])

        grounded = 1.0 if st.is_grounded else 0.0
        colliding = 1.0 if st.is_colliding else 0.0
        falling = 1.0 if self.body.is_falling else 0.0
        pain = self.body.physics_pain / 100.0
        force = min(1.0, st.collision_force / 100.0)
        height_fear = min(1.0, st.position[2] / 5.0)

        sensory = np.concatenate([
            pos / 10.0,          # 3
            vel / 10.0,          # 3
            [spd / 10.0],        # 1
            [up],                # 1
            [grounded],          # 1
            [colliding],         # 1
            [falling],           # 1
            [pain],              # 1
            [force],             # 1
            [height_fear],       # 1
            np.zeros(17),        # pad to 32
        ])
        return sensory.astype(np.float32)


# ---- integrated system ----

class EmbodimentSystem:
    """body + senses + pain feedback, connects to lsm"""

    def __init__(self, body_type="sphere", gui=False, on_pain=None):
        self.body = VirtualBody(
            body_type=body_type, gui=gui,
            on_collision=self._on_collision, on_fall=self._on_fall)
        self.sense = PhysicsSense(self.body)
        self.on_pain = on_pain
        print("embodiment ready")

    def _on_collision(self, ev):
        if self.on_pain:
            self.on_pain(self.body.physics_pain)

    def _on_fall(self, ev):
        print("falling!")

    def spawn(self):
        return self.body.spawn()

    def step(self):
        return self.body.step()

    def get_sensory_input(self):
        return self.sense.get_sensory_vector()

    def get_pain(self):
        return self.body.get_pain()

    def move(self, direction, speed=5.0):
        self.body.move(direction, speed)

    def destroy(self):
        self.body.destroy()


# ---- test ----
if __name__ == "__main__":
    print("=" * 60)
    print("embodiment test")
    print("=" * 60)

    if not HAS_PYBULLET:
        print("no pybullet - pip install pybullet")
    else:
        embody = EmbodimentSystem(gui=True)
        if embody.spawn():
            print("\nbody spawned, running sim...")
            for i in range(240 * 5):
                state = embody.step()
                if i % 240 == 0:
                    print(f"   h={state.position[2]:.2f}m pain={embody.get_pain():.1f}")
                time.sleep(1/240)

            print("\npushing...")
            embody.move("forward", 50)
            for _ in range(240 * 3):
                embody.step()
                time.sleep(1/240)
            embody.destroy()
        print("\ndone")
    print("=" * 60)
