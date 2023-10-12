import torch 
import torch.nn as nn 
import torchvision
import numpy as np
from copy import deepcopy
import warnings
import os
from .filter import *
from .loss import *
from .attacker_base import * 

class GradientAttackerBalanced(GradientAttacker):
    """
    save_last: save the record for the last iteration, as well as the best iteration, including the best for the recon_loss and the smoothness
    """
    def __init__(self, 
            continuous_shapes=None,
            discrete_shapes=None,
            discrete_range=None,
            sequence_index=None,
            init_method='randnposi',
            device=torch.device('cuda'),
            gradient_active=None,
            loss_type='sim', # sim, layerwise_sim
            recon_weight=None,
            lr=0.1,
            betas=(0.9, 0.999),
            momentum=0.9,
            max_iterations=20000,
            milestones=None,
            process_grad=True,
            signed_grad=True,
            error_thre=None,
            record=True,
            save_all=False,
            save_last=True,
            save_best=True,
            save_img=True,
            path='./results/',
            img_path='./images/dqn_',
            img_transpose=True,
            ):
        super().__init__(
            continuous_shapes=continuous_shapes,
            discrete_shapes=discrete_shapes,
            discrete_range=discrete_range,
            sequence_index=sequence_index,
            init_method=init_method,
            device=device,
            gradient_active=gradient_active,
            loss_type=loss_type,
            recon_weight=recon_weight,
            lr=lr,
            betas=betas,
            momentum=momentum,
            max_iterations=max_iterations,
            milestones=milestones,
            process_grad=True,
            signed_grad=True,
            error_thre=error_thre,
            record=record,
            save_all=save_all,
            save_last=save_last,
            save_best=save_best,
            save_img=save_img,
            path=path,
            img_path=img_path,
            img_transpose=img_transpose,)

        self.prior_coeff = 1.
        self.bright_coeff = 0

        self.process_grad = True
        self.signed_grad = True


    def loss_calculation_backup(self, gradient_guess, gradient_label, guess):
        grad_recon = self.losses.recon_loss(gradient_guess, gradient_label)
        img_smoothness = self.losses.smoothness(guess)
        img_bright = self.bright_coeff * self.losses.brightness(guess)
        loss = grad_recon + img_smoothness + img_bright
        log = {}
        if isinstance(grad_recon, torch.Tensor):
            log['recon'] = grad_recon.item()
        else:
            log['recon'] = grad_recon
        if isinstance(img_smoothness, torch.Tensor):
            log['smooth'] = img_smoothness.item()
        else:
            log['smooth'] = img_smoothness
        if isinstance(img_bright, torch.Tensor):
            log['bright'] = img_bright.item()
        else:
            log['bright'] = img_bright
        return loss, log


    def grad_balance(self, guess1, guess2):
        norm1sum = 0
        norm2sum = 0
        with torch.no_grad():
            for g1, g2 in zip(guess1, guess2):
                if g1.grad is not None and g2.grad is not None:
                    norm1 = (g1.grad ** 2).sum()
                    norm2 = (g2.grad ** 2).sum()
            
                    norm1sum += norm1
                    norm2sum += norm2
            
                    norm1 = norm1.sqrt()
                    norm2 = norm2.sqrt()
            
                    g1.grad.data /= norm1 
                    g2.grad.data /= norm2 
                    g1.grad.data += g2.grad.data * self.prior_coeff
                
                    if self.iter % 1000 == 0:
                        print('shape ', g1.shape, 'norm1 ', norm1, ' norm2 ', norm2, ' norm1/norm2 ', norm1/norm2)
        
        if norm1sum == 0 or norm2sum == 0:
            warnings.warn('haven\'t found any tensors with .grad not None')
            return False 
        else:
            return True


    def attack_instance(self, gradient_label, gradient_generator, guess):
        self.iter = 0
        for k in range(self.max_iterations):
            self.optimizer.zero_grad()
            if gradient_generator is None:
                gradient_guess = self.gradient_generator(guess)
            else:
                gradient_guess = gradient_generator(self.shadow_model, guess, self.aux)
            guess_copy = deepcopy(guess)
            loss, log_loss = self.loss_calculation(gradient_guess, gradient_label, guess_copy)
            if torch.isnan(loss):
                warnings.warn('the final loss is nan, the individual losses are {}'.format(log_loss))
                return guess, self.records
            loss.backward()
            self.grad_balance(guess, guess_copy)
            if self.process_grad:
                self.modify_grad(guess)
            self.optimizer.step()
            self.scheduler.step()
            self.prior_constraint(guess)
            if self.records:
                self.records['loss'].append(log_loss)
                self.records['input_guess'].append([guess.detach().cpu().numpy() for guess in guess])
                self.records['iter'].append(k)
                
            if k%10 == 0 or k == self.max_iterations-1:
                if self.records is None:
                    print(k, ' iters, last loss: ', log_loss)
                elif len(self.records['loss']) == 0:
                    print(k, ' iters')
                else:
                    print(k, ' iters, last loss: ', self.records['loss'][-1])
            
            if np.abs(log_loss['recon']) <= self.best_recon_loss:
                self.best_guess = guess
                self.best_gradient_guess = gradient_guess 
                self.best_recon_loss = np.abs(log_loss['recon'])
                
            if True: # this is an image saving paragraph for debugging
                if (k+1)%50 == 0:
                    if self.img_path:
                        self.save_img(params = self.constraint_params)
            
            if k == 10000:
                if self.error_thre is not None:
                    if log_loss['recon'] > self.error_thre:
                        warnings.warn('Early stop at 9999 iteration. This trail is early stopped because it tends to fail. The reason might be a wrong guess of discrete.')
                        break

            self.iter += 1
            
        if self.path:
            if self.save_last:
                last_record = {}
                last_record['loss'] = log_loss
                last_record['input_guess'] = [guess.detach().cpu().numpy() for guess in guess]
                last_record['iter'] = k 
                if self.save_loss:
                    last_record['all_loss'] = self.records['loss'][-self.iter:]
                torch.save(last_record, self.path[0]+'_'+str(self.path[1])+'_last_record.pt') 
            if self.save_all:
                torch.save(self.records, self.path[0]+'_'+str(self.path[1])+'_record.pt') 
            # number indicates how many guess initializations have been tried and stored
            self.path[1] += 1
        if self.img_path and (not k == 10000):
            self.save_img(params=self.constraint_params, name='_last')

        return guess, self.records
