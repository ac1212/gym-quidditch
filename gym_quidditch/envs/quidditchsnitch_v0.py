import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering

class QuidditchSnitchEnv(gym.Env):
    def __init__(self):
        self.pitch_size = 200.0
        self.snitch_mass = 0.25
        self.snitch_max_force = 3.5
        self.snitch_drag_constant = 0.13
        self.seeker_mass = 65.0
        self.seeker_max_force = 455.0
        self.seeker_drag_constant = 8.45
        self.seeker_catch_radius = 1.0
        self.dt = 0.1

        self.action_space = spaces.Box(-1.0,1.0,shape=(2,))
        max_snitch_speed = self.snitch_max_force/self.snitch_drag_constant
        max_seeker_speed = self.seeker_max_force/self.seeker_drag_constant

        ob_space_upper_bound = np.array([
            [self.pitch_size/2,self.pitch_size/2],
            [max_snitch_speed+1.0,max_snitch_speed+1.0],
            [self.pitch_size/2,self.pitch_size/2],
            [max_seeker_speed+1.0,max_seeker_speed+1.0],
            ])
        self.observation_space = spaces.Box(-1*ob_space_upper_bound,ob_space_upper_bound)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        snitch_x, snitch_v, seeker_x, seeker_v = state

        # apply snitch force to snitch
        snitch_f = action*self.snitch_max_force
        snitch_f -= self.snitch_drag_constant*snitch_v
        snitch_a = snitch_f / self.snitch_mass
        snitch_x += snitch_v*self.dt + 0.5*snitch_a*self.dt**2
        snitch_v += snitch_a*self.dt

        # seeker charges towards snitch
        seeker2snitch_x = snitch_x-seeker_x
        distance_seeker2snitch = np.linalg.norm(seeker2snitch_x)
        seeker_f = seeker2snitch_x*self.seeker_max_force/distance_seeker2snitch
        seeker_f -= self.seeker_drag_constant*seeker_v
        seeker_a = seeker_f/self.seeker_mass
        seeker_x += seeker_v*self.dt + 0.5*seeker_a*self.dt**2
        seeker_v += seeker_a*self.dt
        # bounce back if wall hit
        if seeker_x[0]<-self.pitch_size/2: # hit left wall
            seeker_x[0] = -self.pitch_size/2
            seeker_v = np.array([0.0,0.0])
        if seeker_x[0]>self.pitch_size/2: # hit right wall
            seeker_x[0] = self.pitch_size/2
            seeker_v = np.array([0.0,0.0])
        if seeker_x[1]<-self.pitch_size/2: # hit bottom wall
            seeker_x[1] = -self.pitch_size/2
            seeker_v = np.array([0.0,0.0])
        if seeker_x[1]>self.pitch_size/2: # hit top wall
            seeker_x[1] = self.pitch_size/2
            seeker_v = np.array([0.0,0.0])

        # calculate done and reward
        done = np.min(snitch_x)<-self.pitch_size/2 or \
            np.max(snitch_x)>self.pitch_size/2 or \
            distance_seeker2snitch < self.seeker_catch_radius
        reward  = 1.0 if not done else 0.0

        state = np.array((snitch_x, snitch_v, seeker_x, seeker_v))
        return state,reward,done,{}

    def reset(self):
        snitch_start_x,seeker_start_x = self.np_random.uniform(-self.pitch_size/2,self.pitch_size/2,size=(2,2))
        self.state = np.stack((
            snitch_start_x,
            np.array([0.0,0.0]),
            seeker_start_x,
            np.array([0.0,0.0])
        ),axis=0)

    def render(self, mode='human'):
        screen_size = 400
        scale = screen_size/self.pitch_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_size, screen_size)
            self.snitch = rendering.make_circle(radius=self.seeker_catch_radius)
            
            

            self.snitchpos = rendering.Transform()
            self.snitch.add_attr(self.snitchpos)
            self.viewer.add_geom(self.snitch)

            self.seekerpos = rendering.Transform()
            self.seeker = rendering.make_circle(radius=5)
            self.seeker.add_attr(self.seekerpos)
            self.seeker.set_color(1.0,0.7,0.7)
            self.viewer.add_geom(self.seeker)

            self.seeker_range = rendering.make_circle(radius=self.seeker_catch_radius)
            self.seeker_range.add_attr(self.seekerpos)
            self.seeker_range.set_color(0.68,0.0,0.0)
            self.viewer.add_geom(self.seeker_range)

        if self.state is None:
            return None
        
        snitch_x,_,seeker_x,_ = self.state
        self.snitchpos.set_translation(screen_size/2+scale*snitch_x[0],screen_size/2+scale*snitch_x[1])
        self.seekerpos.set_translation(screen_size/2+scale*seeker_x[0],screen_size/2+scale*seeker_x[1])

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




    
