import numpy as np
from physics_sim import PhysicsSim

class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent.

    The task to consider here is Take-off.
    The quadcoputer starts a little above of (x, y, z) = (0, 0, 0).
    The target is set to (x, y, z) = (0, 0, 10.0).

    """
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """

        Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent

        """
        
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0.0
        self.action_high = 900.0
        self.action_size = 4

       
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """
        Uses current pose of sim to return reward and done
        (so that once the quadcopter reaches the height of the target,
        the episode ends. )

        """
        
        reward = - 0.5*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()
        reward += - 2.0*min(abs(self.sim.pose[2] - self.target_pos[2]), 20.0)
        reward +=  4.0*self.sim.v[2]
        
        reward += - 3.0*(abs(self.sim.pose[3:6])).sum()
        reward += - 3.0*(abs(self.sim.angular_v[:3])).sum()

        done = False
        if(self.sim.pose[2] >= self.target_pos[2]):
            reward += 50.0
            done = True

       
        return reward, done

    def step(self, rotor_speeds):
        """
        Uses action to obtain next state, reward, done.

        """
        reward = 0.0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) 
            
            delta_reward, done_height = self.get_reward()
            reward += delta_reward
            if done_height:
                done = done_height
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        return next_state, reward, done

    def reset(self):
        """
        Reset the sim to start a new episode.

        """
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
