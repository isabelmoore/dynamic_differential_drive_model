#!/usr/bin/env python3

import os
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class SaveCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, save_path, verbose=True):
        super(SaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _save(self):
        if self.verbose:
            print('Saving... timesteps: %d' %self.num_timesteps)
        self.model.save(self.save_path+str(self.num_timesteps))

    def _save_best(self):
        # Retrieve training rewards
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            # Calculate mean reward over last episodes
            mean_reward = np.mean(y[-10000:])
            if self.verbose:
                print('timesteps: %d' %self.num_timesteps)
                print('best mean reward: %f, last mean reward: %f'
                        %(self.best_mean_reward, mean_reward))

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose:
                    print('Saving new best model to %s' %self.save_path)
                self.model.save(self.save_path+'_best_reward_'+str(self.num_timesteps))
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self._save()
            self._save_best()
        return True
