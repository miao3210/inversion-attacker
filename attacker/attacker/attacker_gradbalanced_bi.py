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

class GradientAttackerBalancedBipart(GradientAttacker):
    """
    save_last: save the record for the last iteration, as well as the best iteration, including the best for the recon_loss and the smoothness
    """
    def __init__(self, 
            continuous_shapes=None,
            discrete_shapes=None,
            discrete_range=None,
            continuous_index=None,
            gradient_index=None,
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
            save_img=True,
            path='./results/',
            img_path='./images/dqn_',
            img_transpose=True,
            ):
        super().__init__(
            continuous_shapes,
            discrete_shapes,
            discrete_range,
            init_method,
            device,
            gradient_active,
            loss_type,
            recon_weight,
            lr,
            betas,
            momentum,
            max_iterations,
            milestones,
            process_grad=True,
            signed_grad=True,
            error_thre=error_thre,
            record=record,
            save_all=save_all,
            save_last=save_last,
            save_img=save_img,
            path=path,
            img_path=img_path,
            img_transpose=img_transpose,)

        self.continuous_index = continuous_index 
        self.gradient_index = gradient_index

        self.known_continuous = [None for _ in self.continuous_shapes]
        self.known_discrete = None 

        self.prior_coeff = 1.
        self.bright_coeff = 0

        self.process_grad = True
        self.signed_grad = True

        self.feat_coeff = 0.
    
        self.best_recon_idn = []
        self.best_recon_iter = []
        self.best_smooth_idn = []
        self.best_smooth_iter = []

    def feat_loss(self, feat_conv, feat_fc):
        loss = nn.MSELoss()
        return loss(feat_conv, feat_fc)

    def gradient_generator(self, guess, phase):
        guess_input = []
        for g in guess:
            if len(g.shape) == 4:
                if self.use_act:
                    g = self.act(g)
            guess_input.append(g)
        if phase in ['fc', 'both']:
            self.shadow_model.zero_grad()
            output, feat = self.shadow_inference(self.shadow_model, guess_input, phase=phase)
            loss = self.shadow_loss(output, guess)
            gradient_guess = torch.autograd.grad(loss, self.shadow_model.parameters(),
                                create_graph=True, allow_unused=True) #, retain_graph=True) 
        elif phase in ['conv']:
            self.shadow_model.zero_grad()
            output, feat = self.shadow_inference(self.shadow_model, guess_input, phase=phase)
            gradient_guess = torch.autograd.grad(feat, self.shadow_model.parameters(),
                                grad_outputs=self.guesses[2][1], create_graph=True, allow_unused=True)
        grad_guess = []
        for g in gradient_guess:
            if g is not None:
                grad_guess.append(g)
        return grad_guess, feat


    def loss_calculation(self, gradient_guess, gradient_label, guess, feat, phase):
        grad_recon = self.losses.recon_loss(gradient_guess, gradient_label)
        img_smoothness = self.losses.smoothness(guess)
        img_bright = self.bright_coeff * self.losses.brightness(guess)
        log = {}
        loss = 0
        if phase in ['fc', 'conv', 'both']:
            loss += grad_recon
            if isinstance(grad_recon, torch.Tensor):
                log['recon'] = grad_recon.item()
            else:
                log['recon'] = grad_recon
        if phase in ['conv', 'both']:
            loss += (img_smoothness + img_bright)
            if isinstance(img_smoothness, torch.Tensor):
                log['smooth'] = img_smoothness.item()
            else:
                log['smooth'] = img_smoothness
            if isinstance(img_bright, torch.Tensor):
                log['bright'] = img_bright.item()
            else:
                log['bright'] = img_bright
        elif phase in ['conv']:
            feat_brk = self.feat_coeff * self.feat_loss(feat, self.guesses[2][0])
            loss += feat_brk
            if isinstance(feat_brk, torch.Tensor):
                log['brkpoint'] = feat_brk.item()
            else:
                log['brkpoint'] = feat_brk
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


    def set_up_prior_knowledge_based_refine(self, prior_knowledge_based_refine):
        ## this is used to further process some estimated data and use them as known, i.e. fixed, no more optimization
        self.prior_knowledge_based_refine = prior_knowledge_based_refine


    def attack_instance(self, gradient_label, gradient_generator, guess, constraint_options, constraint_params, phase):

        self.iter = 0
        for k in range(self.max_iterations):
            self.optimizer.zero_grad()
            if gradient_generator is None:
                gradient_guess, feat = self.gradient_generator(guess, phase=phase)
            else:
                gradient_guess, feat = gradient_generator(guess, phase=phase)
            guess_copy = deepcopy(guess)
            loss, log_loss = self.loss_calculation(gradient_guess, gradient_label, guess_copy, feat, phase=phase)
            if torch.isnan(loss):
                warnings.warn('the final loss is nan, the individual losses are {}'.format(log_loss))
                return guess, self.records
            loss.backward()
            if phase in ['conv', 'both']:
                self.grad_balance(guess, guess_copy)
            if self.process_grad:
                self.modify_grad(guess)
            self.optimizer.step()
            self.scheduler.step()
            self.prior_constraint(guess, constraint_options, constraint_params)
            if self.records:
                self.records['loss'].append(log_loss)
                self.records['input_guess'].append([guess.detach().cpu().numpy() for guess in guess])
                self.records['iter'].append(k)
                
            if np.abs(log_loss['recon']) <= self.best_recon_loss:
                self.best_guess = guess
                self.best_gradient_guess = gradient_guess 
                self.best_recon_loss = np.abs(log_loss['recon'])
                
            if k%1000 == 0 or k == self.max_iterations-1:
                if self.records is None:
                    print(k, ' iters, last loss: ', log_loss)
                elif len(self.records['loss']) == 0:
                    print(k, ' iters')
                else:
                    print(k, ' iters, last loss: ', self.records['loss'][-1])
            
            if True: # this is an image saving paragraph for debugging
                if phase in ['conv', 'both']:
                    if (k+1)%5000 == 0:
                        if self.img_path:
                            self.save_img(params = self.constraint_params)
            
            if k == 10000:
                if self.error_thre is not None:
                    if log_loss['recon'] > self.error_thre:
                        warnings.warn('Early stop at 9999 iteration. This trail is early stopped because it tends to fail. The reason might be a wrong guess of discrete.')
                        break

            self.iter += 1
            
        ## locate the best recon: 
        ## iter is the index within this attack instance, 
        ## while idn is the index in the record dict, which contains records for all 3 phases
        recon_losses = np.array([r['recon'] for r in self.records['loss'][-self.iter:]]) # only check on the record of this instance
        zero_iter = (recon_losses==0).nonzero()[0]
        if len(zero_iter) > 0:
            best_recon_iter = zero_iter[-1]
        else:
            best_recon = np.abs(recon_losses).min()
            print('ready to locate the min recon loss in attacker')
            best_recon_iter = (recon_losses == best_recon).nonzero()[0][-1]
        self.best_recon_iter[-1][phase] = best_recon_iter
        best_recon_idn = best_recon_iter - self.iter # negative index
        self.best_recon_idn[-1][phase] = len(self.records['loss']) + best_recon_idn # positive index
        
        if phase == 'fc':
            ## calculate the corresponding feature_brkpoint and save in self.guesses
            best_recon_guess = self.records['input_guess'][best_recon_idn]
            brkpoint = [torch.tensor(b, device=self.device) for b in best_recon_guess] #self.guess[0].detach()
            feature_brkpoint = brkpoint[1]
            feature_brkpoint.requires_grad = True
            output, _ = self.shadow_inference(self.shadow_model, brkpoint, phase=phase)
            loss = self.shadow_loss(output, guess)
            derivative_brkpoint = torch.autograd.grad(loss, feature_brkpoint)
            self.guesses.append([feature_brkpoint.detach(), derivative_brkpoint[0].detach()])
            known = self.prior_knowledge_based_refine(guess=self.records['input_guess'][best_recon_idn])
            self.oracle(known)

        elif phase in ['conv', 'both']:
            if self.img_path:
                # save img as last
                if not k == 10000:
                    self.save_img(params=self.constraint_params, name='_last_'+phase+'.jpg')
                # save img as best recon
                best_recon_guess = self.records['input_guess'][best_recon_idn]
                self.save_img(guess=best_recon_guess, params=self.constraint_params, name='_best_recon_'+str(best_recon_iter)+'_'+phase+'.jpg')
                # save img as best smooth
                best_smooth_iter = np.array([r['smooth'] for r in self.records['loss'][-self.iter:]]).argmin()
                self.best_smooth_iter[-1][phase] = best_smooth_iter
                best_smooth_idn = best_smooth_iter - self.iter # negative index
                self.best_smooth_idn[-1][phase] = len(self.records['loss']) + best_smooth_idn # positive index
                best_smooth_guess = self.records['input_guess'][best_smooth_idn]
                self.save_img(guess=best_smooth_guess, params=self.constraint_params, name='_best_smooth_'+str(best_recon_iter)+'_'+phase+'.jpg')

        if self.path:
            # save records
            if self.save_last:
                last_record = {}
                last_record['loss'] = log_loss
                last_record['input_guess'] = [guess.detach().cpu().numpy() for guess in guess]
                last_record['iter'] = k 
                if self.save_loss:
                    last_record['all_loss'] = self.records['loss'][-self.iter:]
                torch.save(last_record, self.path[0]+'_'+str(self.path[1])+'_'+phase+'_last_record.pt') 
                self.path[1] += 1
            
            if self.save_best:
                best_recon_record = {}
                for k in self.records.keys():
                    if k == 'gradient_label':
                        best_recon_record[k] = self.records[k]
                    else:
                        best_recon_record[k] = self.records[k][best_recon_idn]
                torch.save(best_recon_record, self.path[0]+'_'+str(best_recon_iter)+'_'+str(self.path[1])+'_'+phase+'_best_recon_record.pt') 
                self.path[1] += 1
                
                if phase in ['conv', 'both']:
                    best_smooth_record = {}
                    for k in self.records.keys():
                        if k == 'gradient_label':
                            best_smooth_record[k] = self.records[k]
                        else:
                            best_smooth_record[k] = self.records[k][best_smooth_idn]
                    torch.save(best_smooth_record, self.path[0]+'_'+str(best_smooth_iter)+'_'+str(self.path[1])+'_'+phase+'_best_smooth_record.pt') 
                    self.path[1] += 1
                
            if self.save_all:
                torch.save(self.records, self.path[0]+'_'+str(self.path[1])+'_ '+phase+'_record.pt') 
            # number indicates how many guess initializations have been tried and stored
            self.path[1] += 1
        
        return guess, self.records



    def attack(self, gradient_label, gradient_generator=None, bsz1=False, case=None, last_idn=None): 
        self.guesses_init(continuous_shapes=None, discrete_shapes=None, discrete_range=None, disable_discrete_init=False, milestones=self.milestones[0]) # only initialize the discrete variables
        if len(self.guesses[1]) == 0:
            # when all guess_data are continuous
            self.best_recon_idn.append({})
            self.best_recon_iter.append({})
            self.best_smooth_idn.append({})
            self.best_smooth_iter.append({})
            for phase, index, gradindex, milestone in zip(['fc', 'conv', 'both'], self.continuous_index, self.gradient_index, self.milestones):
                if phase == 'both':
                    warnings.warn('current attack assumes that all the continuous variables optimized in the "conv" phase need to be optimized in the "both" phase')
                    prior_initialization = [g for g in self.guesses[0]]
                else:
                    prior_initialization = None
                self.guesses_init(self.continuous_shapes[index], discrete_shapes=None, discrete_range=None, known_continuous=self.known_continuous[index], prior_initialization=prior_initialization, milestones=milestone)
                guess = self.guesses[0]
                constraint_options = self.constraint_options[index]
                constraint_params = self.constraint_params[index]
                self.attack_instance(gradient_label[gradient_index], gradient_generator, guess, phase=phase)
        else:
            if bsz1:
                # the identification of discrete data is possible only when the batchsize is 1
                assert case in ['classification', 'rl', 'unavailable'] # this is to help identify the label for the discrete data guess; unavailable refers to the direct conclusion about the discrete data is unavailable
                if case in ['classification', 'rl']:
                    assert isinstance(last_idn, int), 'last_idn is used to identify the parameter that reveals the discrete label or action, which has to be integer'
            # when discrete data exists
            if case in ['classification', 'rl']:
                assert isinstance(last_idn, int)
                warnings.warn('current attack assume the first discrete guess is the one that can be identified by the gradient of last layer')
                last_grad = gradient_label[last_idn]
                if case == 'classification':
                    assert len(last_grad[last_grad<0]) == 1
                    gd_last_idn = torch.tensor(torch.argmin(last_grad)).view(-1) * torch.ones_like(self.guesses[1][0][0])
                    start = 1
                elif case == 'rl':
                    assert len(last_grad[last_grad!=0]) == 1
                    gd_last_idn = torch.nonzero(last_grad).view(-1) * torch.ones_like(self.guesses[1][0][0])
                    err_direc = last_grad.sum().sign() * (-100) * torch.ones_like(self.guesses[1][1][0])
                    start = 2
                    print('Identified the action {} and the direction of expected Q value {}'.format(gd_last_idn, err_direc))
                else:
                    gd_last_idn = None
                    start = 0
            else:
                start = 0
            # count all the combinations for discrete guesses
            discrete_counts = []
            for k,gd in enumerate(self.guesses[1]):
                if k<start:
                    continue
                discrete_counts.append(torch.arange(len(gd)))
            # enumerate all the discrete data guesses
            if len(discrete_counts) == 0:
                self.best_recon_idn.append({})
                self.best_recon_iter.append({})
                self.best_smooth_idn.append({})
                self.best_smooth_iter.append({})
                # when there is only discrete data that is identified by the last_grad
                for phase, index, gradindex, milestone in zip(['fc', 'conv', 'both'], self.continuous_index, self.gradient_index, self.milestones):
                    if phase == 'both':
                        warnings.warn('current attack assumes that all the continuous variables optimized in the "conv" phase need to be optimized in the "both" phase')
                        prior_initialization = [g for g in self.guesses[0]]
                    else:
                        prior_initialization = None
                    self.guesses_init(self.continuous_shapes[index], discrete_shapes=None, discrete_range=None, disable_discrete_init=True, known_continuous=self.known_continuous[index], prior_initialization=prior_initialization, milestones=milestone)
                    guess = self.guesses[0]
                    if gd_last_idn is not None:
                        guess.append(gd_last_idn)
                        guess.append(err_direc)
                    constraint_options = self.constraint_options[index]
                    constraint_params = self.constraint_params[index]
                    self.attack_instance(gradient_label[gradindex], gradient_generator, guess, constraint_options, constraint_params, phase=phase) 
            else:
                # enumerate over all the combinations
                discrete_combs = torch.cartesian_prod(*discrete_counts) # all possible combinations
                if len(discrete_combs.shape) == 1:
                    discrete_combs = discrete_combs[:,None]
                for discrete_idn in discrete_combs:
                    self.best_recon_idn.append({})
                    self.best_recon_iter.append({})
                    self.best_smooth_idn.append({})
                    self.best_smooth_iter.append({})
                    for phase, index, gradindex, milestone in zip(['fc', 'conv', 'both'], self.continuous_index, self.gradient_index, self.milestones):
                        if phase == 'both':
                            warnings.warn('current attack assumes that all the continuous variables optimized in the "conv" phase need to be optimized in the "both" phase')
                            prior_initialization = [g for g in self.guesses[0]]
                        else:
                            prior_initialization = None
                        self.guesses_init(self.continuous_shapes[index], discrete_shapes=None, discrete_range=None, disable_discrete_init=True, known_continuous=self.known_continuous[index], prior_initialization=prior_initialization, milestones=milestone)
                        guess = self.guesses[0]
                        if gd_last_idn is not None:
                            guess.append(gd_last_idn)
                            guess.append(err_direc)
                        if len(self.guesses[1][start:])>0:
                            for gd, idn in zip(self.guesses[1][start:], discrete_idn):
                                guess.append(gd[idn])
                        constraint_options = self.constraint_options[index]
                        constraint_params = self.constraint_params[index]
                        self.attack_instance(gradient_label[gradindex], gradient_generator, guess, constraint_options, constraint_params, phase=phase) 

        if self.records is not None:
            if len(self.records['loss']) > 0:
                smooth_losses = []
                for r in self.records['loss']:
                    if 'smooth' in r.keys():
                        smooth_losses.append(r['smooth'])
                best_smooth_idn = np.array(smooth_losses).argmin() - len(smooth_losses) # negative index
                self.best_smooth_idn.append( best_smooth_idn )
                best_smooth_iter = self.records['iter'][best_smooth_idn]
                self.best_smooth_iter.append( best_smooth_iter )
                if self.path:
                    if self.save_all:
                        torch.save(self.records, self.path[0]+'_'+str(self.path[1])+'_record.pt') 
                        # number indicates how many guess initializations have been tried and stored
                        self.path[1] += 1

                if self.img_path:
                    best_smooth_guess = self.records['input_guess'][best_smooth_idn]
                    self.save_img(guess=best_smooth_guess, params=self.constraint_params, name='_acrossphase_best_smooth_'+str(best_smooth_iter)+'.jpg')
                    
