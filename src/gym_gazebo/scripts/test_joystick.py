#!/usr/bin/env python3

import threading
import time
import sys
import tty
import termios
import getopt

from gym_env import MobileRobotEnv

class KeyReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)

    def __del__(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        print()

    def read(self):
        ch = sys.stdin.read(1)
        return ch

class Joystick:
    def __init__(self, key_reader, action_range=[(-1, 1), (-1, 1)],
            increment=(0.1, 0.1), nominal=(-1.0, 0.0)):
        self.input = key_reader
        self.action_range = action_range
        self.increment = increment
        self.output = list(nominal)
        self.nominal = nominal
        self.pause = False
        self.quit = False
        self.shutdown = False
        self.lock = threading.Lock()

    def run(self):
        while True:
            input_ = self.input.read()
            with self.lock:
                if input_ == ',':
                    self.output[0] = max(self.output[0] - self.increment[0],
                            self.action_range[0][0])
                elif input_ == 'i':
                    self.output[0] = min(self.output[0] + self.increment[0],
                            self.action_range[0][1])
                elif input_ == 'j':
                    self.output[1] = max(self.output[1] - self.increment[1],
                            self.action_range[1][0])
                elif input_ == 'l':
                    self.output[1] = min(self.output[1] + self.increment[1],
                            self.action_range[1][1])
                elif input_ == 'k':
                    self.output = list(self.nominal)
                elif input_ == 'p':
                    self.pause = not self.pause
                elif input_ == 'q' or input_ == '\x03':
                    self.quit = True
                if self.shutdown:
                    return

def main():
    usage = 'usage: %s [-i, --infinite]' %sys.argv[0]
    
    infinite = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i', ['infinite'])
        for o, a in opts:
            if o in ('-i', '--infinite'):
                infinite = True
    except getopt.GetoptError:
        print(usage)
        return

    # Create environment
    env = MobileRobotEnv(max_episode_steps=300, bound=3.0,
            goal_distance=3.0, goal_radius=0.2, timestep=0.05,
            use_continuous_actions=True, use_shaped_reward=True, use_model=True,
            evaluate=False, logfile='test.log', debug=True)

    # Start joystick
    js = Joystick(KeyReader())
    t = threading.Thread(target=js.run)
    t.start()

    # Run episode
    goal = (3.0, 0) # x, y
    action = (0.0, 0.0) # lx, az

    done = False
    obs = env.reset()
    env.set_goal(goal)
    episode_reward = 0
    while not done:
        # update action
        with js.lock:
            if js.quit:
                break
            action = js.output # lx, az

        # step
        obs, reward, done, info = env.step(action)
        if infinite:
            done = False
        episode_reward += reward
        
        # sleep
        time.sleep(0.0)
    
    print('episode reward: %f, steps: %d' %(episode_reward, env.episode_steps))
    js.shutdown = True
    t.join()

if __name__ == '__main__':
    main()
