import numpy as np
from abc import ABC, abstractmethod


class RandomWalkMotionModel:
    def __init__(self, seed=None, theta_sigma=0.1, v_max=0.2):
        self.rng = np.random.default_rng(seed)
        self.theta_sigma = theta_sigma
        self.v_max = v_max
        self.reset()

    def reset(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = 0.0
        self.speed = 0.0

    def _sample_theta(self):
        if self.theta == 0:
            self.theta = self.rng.uniform(-np.pi, np.pi)

        self.theta += self.rng.normal(0.0, self.theta_sigma)
        # self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def _sample_speed(self):
        self.speed = self.rng.uniform(0.0, self.v_max)

    def propose_step(self):
        self._sample_theta()
        self._sample_speed()

        dx = self.speed * np.cos(self.theta)
        dy = self.speed * np.sin(self.theta)

        return dx, dy, self.theta, self.speed
    

class Arena(ABC):
    @abstractmethod
    def contains(self, x, y):
        pass

    def center(self):
        pass
    
    def clamp(self, x, y):
        raise NotImplementedError
    
    
class SquareArena(Arena):
    def __init__(self, size, eps=1e-6):
        if size <= 0:
            raise ValueError("Arena size must be positive.")
        
        self.size = float(size)
        self._half_size = size / 2.0
        self.eps = float(eps)

    def contains(self, x, y):
        return (
            x >= - (self._half_size + self.eps)
            ) & (
                x <= self._half_size + self.eps
                ) & (
                    y >= - (self._half_size + self.eps)
                    ) & (
                        y <= self._half_size + self.eps
                        )
    
    def center(self):
        return 0.0, 0.0


class SimulationController:
    def __init__(self, motion_model, arena, max_tries=100):
        self.motion_model = motion_model
        self.arena = arena
        self.max_tries = int(max_tries)

        self.reset()
    
    def reset(self):
        self.t = 0
        self.data = []

        x0, y0 = self.arena.center()
        self.motion_model.reset(x=x0, y=y0)

    def _apply_step(self, x, y, theta, speed):
        self.t += 1
        self.motion_model.x = x
        self.motion_model.y = y
        self.motion_model.theta = theta
        self.motion_model.speed = speed

    def _record_step(self):
        record = {
                "t": self.t,
                "theta": self.motion_model.theta,
                "speed": self.motion_model.speed,
                "x": self.motion_model.x,
                "y": self.motion_model.y,
        }

        self.data.append(record)
        return record

    def step(self):
        x0, y0 = self.motion_model.x, self.motion_model.y

        for _ in range(self.max_tries):
            dx, dy, theta_candidate, speed_candidate = self.motion_model.propose_step()
            x_candidate = x0 + dx
            y_candidate = y0 + dy

            if self.arena.contains(x_candidate, y_candidate):
                x1, y1 = x_candidate, y_candidate
                theta, speed = theta_candidate, speed_candidate
                break
        else:
            x1, y1 = x0, y0
            theta = self.motion_model.theta
            speed = 0.0

        self._apply_step(x1, y1, theta, speed)

        return self._record_step()

    def generate(self, num_steps):
        self.reset()
        self._record_step()

        for _ in range(1, num_steps):
            self.step()
        
        return self.data