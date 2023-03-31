import numpy as np
import torch

class EarlyStopping:
    def __init__(self, num_machines, patience=10, model_save_path = ''):
        self.patience = patience
        self.counter = 0
        self.best_score_g = None
        self.best_score_l = None
        self.early_stop = False
        self.model_save_path = model_save_path

    def step(self, acc_l, phase, model, acc_g=None):
        if phase == 0:
            score = acc_g
            if self.best_score_g is None:
                self.best_score_g = score
                self.save_checkpoint(model,phase)
            elif score < self.best_score_g:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} got {score} while best is {self.best_score_g}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score_g = score
                self.save_checkpoint(model,phase)
                self.counter = 0
            self.best_score_l = acc_l
        else:
            score = acc_l
            if self.best_score_l is None:
                self.best_score_l = score
                self.save_checkpoint(model,phase)
            elif score < self.best_score_l:
                self.counter += 1
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience} got {score} while best is {self.best_score_l}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score_l = score
                self.save_checkpoint(model,phase)
                self.counter = 0
        return self.early_stop
    
    def reset(self):
        self.counter = 0
        self.early_stop = False

    def save_checkpoint(self, model, phase):
        '''Saves model when validation loss decrease.'''
        print("Saving model")
        if phase:
            torch.save(model.state_dict(), self.model_save_path)
        else:
            torch.save(model.module.state_dict(), self.model_save_path)
