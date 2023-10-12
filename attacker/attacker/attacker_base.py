import torch 
import torch.nn as nn 
import torchvision
import numpy as np
from copy import deepcopy
import warnings
import os
from .filter import *
from .loss import *

class GradientAttacker():
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
            save_loss=False,
            save_img=True,
            path='./results/',
            img_path='./images/dqn_',
            img_transpose=False,
            smooth_coeff=1e-1,
            bright_coeff=0,
            sequence_smooth_coeff=1e-1,
            ):
        self.continuous_shapes = continuous_shapes
        self.discrete_shapes = discrete_shapes
        self.discrete_range = discrete_range
        if self.discrete_shapes is None or self.discrete_range is None:
            print('no discrete variable will be initialized')
            self.discrete_shapes = []
            self.discrete_range = []
        self.sequence_index = sequence_index
        self.init_method = init_method

        self.gradient_active = gradient_active
        self.loss_type = loss_type
        self.use_act = False
        self.losses = Loss(gradient_active=gradient_active,
                    loss_type=loss_type,
                    use_act=self.use_act,
                    device=device)
        
        if self.sequence_index is not None:
            self.losses.set_sequence_loss(self.sequence_index)

        if self.use_act:
            self.losses.act = self.act

        self.lr = lr 
        self.betas = betas 
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.milestones = milestones
        self.device = device
        self.constraint_options = []
        self.constraint_params = []
        self.known_continuous = None 
        self.known_discrete = None 

        if isinstance(self.milestones, list):
            if not isinstance(self.milestones[0], list): # this is to avoid failure in balancedbi attacker
                self.guesses_init(continuous_shapes, discrete_shapes, discrete_range)
        else:
            self.guesses_init(continuous_shapes, discrete_shapes, discrete_range)
        
        self.process_grad = process_grad
        self.signed_grad = signed_grad
        self.clip_grad = False

        self.smooth_coeff = smooth_coeff# 2e-2 works better 
        self.bright_coeff = bright_coeff
        self.sequence_smooth_coeff = sequence_smooth_coeff

        self.sig = nn.Sigmoid()

        self.filter_median = MedianPool2d()
        self.filter_mask = Mask()

        self.print_freq = 100 # 1000
        self.save_img_freq = 2000 # 5000

        def _check_path(path):
            folder_path = os.path.dirname(path)
            if len(folder_path) > 0:
                if not os.path.exists(folder_path):
                    print('create folder: ' + folder_path)
                    os.mkdir(folder_path)
            else:
                print('the path is in the current directory')

        if save_img:
            #record = True
            _check_path(img_path)
            self.img_path = [img_path, 0]
        else:
            self.img_path = None

        self.img_transpose = img_transpose
        self.img_name_generator = None

        self.save_all = save_all
        self.save_last = save_last
        self.save_best = save_best
        self.save_loss = save_loss 
        if save_all:
            record = True

        self.path = None
        self.records = None
        if record:
            self.clear_records()
            if path is not None and (self.save_all or self.save_best or self.save_last or self.save_loss):
                _check_path(path)
                self.path = [path, 1]             
        
        self.error_thre = error_thre

        self.best_guess = None
        self.best_gradient_guess = None
        self.best_recon_loss = 9e9

        self.aux = None

        self.nan_occur = False

    def act(self, x):
        return self.sig(x)

    def clear_records(self,):
        self.records = {'loss':[], 
                        'input_guess':[], 
                        'iter':[], 
                        'gradient_label': None}


    def loss_calculation(self, gradient_guess, gradient_label, guess):
        grad_recon = self.losses.recon_loss(gradient_guess, gradient_label)
        if torch.is_tensor(self.smooth_coeff):
            img_smoothness = (self.smooth_coeff * self.losses.smoothness_independent(guess)).sum()
        else:
            img_smoothness = self.smooth_coeff * self.losses.smoothness(guess)
        if torch.is_tensor(self.bright_coeff):
            img_bright = (self.bright_coeff * self.losses.brightness_independent(guess)).sum()
        else:
            img_bright = self.bright_coeff * self.losses.brightness(guess)
        if torch.is_tensor(self.sequence_smooth_coeff):
            sequence_smoothness = (self.sequence_smooth_coeff * self.losses.sequence_smoothness_independent(guess)).sum()
        else:
            sequence_smoothness = self.sequence_smooth_coeff * self.losses.sequence_smoothness(guess)
        loss = grad_recon + img_smoothness + img_bright + sequence_smoothness
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
        if self.sequence_index is not None:
            if isinstance(sequence_smoothness, torch.Tensor):
                log['sequence'] = sequence_smoothness.item()
            else:
                log['sequence'] = sequence_smoothness
        return loss, log


    def modify_grad(self, guess):
        for g in guess:
            if g.grad is not None and len(g.shape)==4:
                if self.signed_grad:
                    g.grad.data = g.grad.data.sign()
                elif self.clip_grad:
                    g.grad.data = torch.clamp(g.grad.data, min=-1, max=1)
                else:
                    raise RuntimeError('something about the modify_grad, check the code')


    def set_up_prior_constraint(self, options=None, params=None):
        assert len(options) == len(params), 'The constraint options has {} elements, but the constraint params has {} elements'.format(len(options), len(params))
        if options is not None:
            if len(options) > 0:
                self.constraint_options = options
                self.losses.constraint_options = options
        if params is not None:
            if len(params) > 0:
                self.constraint_params = params


    def prior_constraint(self, guess, constraint_options=None, constraint_params=None):
        if constraint_options is None:
            constraint_options = self.constraint_options
        if constraint_params is None:
            constraint_params = self.constraint_params
        if len(constraint_options) == 0:
            return 'no prior constraint has been applied'
        else:
            assert len(guess) >= len(constraint_options), 'Found more constraint options({}) than data to guess({})'.format(len(self.constraint_options), len(guess))
            for option, param, g in zip(constraint_options, constraint_params, guess):
                with torch.no_grad():
                    if option == 'image':
                        if len(param) == 4:
                            dataset_mean, dataset_std, low, up = param
                            g.data = torch.clamp(g, min=(low-dataset_mean)/dataset_std, max=(up-dataset_mean)/dataset_std)
                        elif len(param) == 2:
                            low, up = param 
                            g.data = torch.clamp(g, min=low, max=up)
                        else:
                            raise RuntimeError('The constraint param for box constraint has to have 2 or 4 elements, but got {}'.format(param))
                        
                    elif option == 'mask':
                        continue
                        # this doesn't work
                        assert len(param) == 2
                        low, up = param 
                        g.data = torch.clamp(g, min=low, max=up)
                        g.data = self.filter_median(g.data)
                        if self.iter % 100 == 0:
                            mid = (low + up)/2
                            g.data[g>mid] = up
                            g.data[g<=mid] = low
                        #if self.iter > 1 and self.iter % 5000 == 0:   
                        if self.iter == 10000:
                            g.data = self.filter_mask(g.data)
                            g.requires_grad = False
                    elif option == 'box':
                        assert len(param) == 2
                        low, up = param 
                        g.data = torch.clamp(g, min=low, max=up)
                    elif option is None:
                        pass
                    else:
                        raise NotImplementedError


    def guesses_init(self, continuous_shapes=None, discrete_shapes=None, discrete_range=None, disable_discrete_init=False, 
                        lr=None, max_iterations=None, known_continuous=None, prior_initialization=None, milestones=None):
        if continuous_shapes is None:
            continuous_shapes = self.continuous_shapes
        if discrete_shapes is None:
            discrete_shapes = self.discrete_shapes
        if discrete_range is None:
            discrete_range = self.discrete_range
        if milestones is None:
            milestones = self.milestones

        assert len(self.discrete_shapes) == len(self.discrete_range)
        if lr is None:
            assert isinstance(self.lr, float), 'when lr is not given, self.lr has to be given as float'
            lr = self.lr
        if max_iterations is None:
            max_iterations = self.max_iterations 

        if disable_discrete_init is False:
            guesses = [[], []]
            guesses[1] = [[] for _ in range(len(discrete_shapes))]
            
            for k,guess_discrete in enumerate(guesses[1]):
                if len(discrete_shapes[k]) > 1 or discrete_shapes[k][-1] > 1:
                    warnings.warn('current discrete data guess implementation only support the scalar case, check the GradientAttacker for detail')
                flag = True
                if self.known_discrete is not None:
                    if self.known_discrete[k] is not None:
                        guesses[1][k].append((torch.ones(discrete_shapes[k], dtype=int).to(self.device)*torch.tensor(self.known_discrete[k], device=self.device)).detach()) 
                        flag = False 
                if flag:
                    if len(discrete_range[k]) == 2:
                        for v in range(discrete_range[k][0], discrete_range[k][1]):
                            guesses[1][k].append((torch.ones(discrete_shapes[k], dtype=int).to(self.device)*v).detach()) 
                    elif len(discrete_range[k]) == 3:
                        for v in range(discrete_range[k][0], discrete_range[k][1], discrete_range[k][2]):
                            guesses[1][k].append((torch.ones(discrete_shapes[k], dtype=int).to(self.device)*v).detach()) 
        else:
            guesses = self.guesses
            guesses[0] = []

        if self.init_method == 'randn':
            for k,s in enumerate(continuous_shapes):
                g = torch.randn(s).to(self.device) * 0.01
                if isinstance(prior_initialization, list):
                    if prior_initialization[k] is not None:
                        g = prior_initialization[k]
                guesses[0].append(nn.Parameter(g.detach(), requires_grad=True))
        elif self.init_method == 'randnposi':
            for k,s in enumerate(continuous_shapes):
                g = torch.randn(s).abs().to(self.device) * 0.01
                if isinstance(prior_initialization, list):
                    if prior_initialization[k] is not None:
                        g = prior_initialization[k]
                guesses[0].append(nn.Parameter(g.detach(), requires_grad=True))
        elif self.init_method == 'zero':
            for k,s in enumerate(continuous_shapes):
                g = torch.zeros(s).to(self.device)
                if isinstance(prior_initialization, list):
                    if prior_initialization[k] is not None:
                        g = prior_initialization[k]
                guesses[0].append(nn.Parameter(g.detach(), requires_grad=True))
        elif self.init_method == 'one':
            for k,s in enumerate(continuous_shapes):
                g = torch.ones(s).to(self.device)
                if isinstance(prior_initialization, list):
                    if prior_initialization[k] is not None:
                        g = prior_initialization[k]
                guesses[0].append(nn.Parameter(g.detach(), requires_grad=True))
        else:
            raise NotImplementedError

        self.guesses = guesses
        if known_continuous is None:
            known_continuous = self.known_continuous
        if known_continuous is not None:
            assert len(known_continuous) == len(self.guesses[0]), 'found {} continuous data in guesses, but {} known variables'.format(len(self.guesses[0]), len(known_continuous))
            for kn, g in zip(known_continuous, self.guesses[0]):
                if kn is not None:
                    g.data = torch.tensor(kn).to(self.device) #* torch.ones(g.shape, device=self.device)
                    g.requires_grad = False
        if len(self.guesses[0]) > 0:
            self.optimizer = torch.optim.AdamW(self.guesses[0], lr=lr, betas=self.betas)
            #self.optimizer = torch.optim.SGD(self.guesses[0], lr=lr)
            if milestones is None:
                milestones = [max_iterations // 10, max_iterations // 5, max_iterations // 2, max_iterations // 1.25]
                                    #milestones=[max_iterations // 5, max_iterations // 1.9231, max_iterations // 1.1905], # 5k, 13k, 21k, (25k)
                                    #[max_iterations // 2.667, max_iterations // 1.6, max_iterations // 1.142], 
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                    milestones=milestones, # 4k, 10k, 16k, (20k)
                                    gamma=0.1)
        else:
            self.optimizer = None 
            self.scheduler = None
        #self.optimizer = torch.optim.SGD(self.guesses[0], lr=self.lr, momentum=self.momentum)


    def capture_gradient(self, models):
        gradient_label = []
        if isinstance(models, list):
            for m in models:
                for p in m.parameters():
                    if p.grad is not None:
                        gradient_label.append(p.grad)
        elif isinstance(models, nn.Module):
            for p in models.parameters():
                if p.grad is not None:
                    gradient_label.append(p.grad)
        return gradient_label


    def attack(self, gradient_label, gradient_generator=None, bsz1=False, case=None, last_idn=None, 
                    continuous_shapes=None, discrete_shapes=None, discrete_range=None): 
        self.guesses_init(continuous_shapes, discrete_shapes, discrete_range, disable_discrete_init=False)
        if len(self.guesses[1]) == 0:
            # when all guess_data are continuous
            self.guesses_init(continuous_shapes, discrete_shapes, discrete_range)
            guess = self.guesses[0]
            self.attack_instance(gradient_label, gradient_generator, guess)
        else:
            if bsz1:
                # the identification of discrete data is possible only when the batchsize is 1
                assert case in ['classification', 'rl', 'mse', 'unavailable'] # this is to help identify the label for the discrete data guess; unavailable refers to the direct conclusion about the discrete data is unavailable
                if case in ['classification', 'rl']:
                    assert isinstance(last_idn, int), 'last_idn is used to identify the parameter that reveals the discrete label or action, which has to be integer'
            # when discrete data exists
            if bsz1 and ( case in ['classification', 'rl', 'mse'] ):
                assert isinstance(last_idn, int)
                warnings.warn('current attack assume the first discrete guess is the one that can be identified by the gradient of last layer')
                last_grad = gradient_label[last_idn]
                if case == 'classification':
                    assert len(last_grad[last_grad<0]) == 1
                    gd_last_idn = torch.tensor(torch.argmin(last_grad)).view(-1) * torch.ones_like(self.guesses[1][0][0])
                    start = 1
                    err_direc = None
                elif case == 'rl':
                    assert len(last_grad[last_grad!=0]) == 1
                    gd_last_idn = torch.nonzero(last_grad).view(-1) * torch.ones_like(self.guesses[1][0][0])
                    err_direc = last_grad.sum().sign() * (-100) * torch.ones_like(self.guesses[1][1][0])
                    start = 2
                    print('Identified the action {} and the direction of expected Q value {}'.format(gd_last_idn, err_direc))
                elif case == 'mse':
                    gd_last_idn = None
                    err_direc = last_grad.sum().sign() * (-100) * torch.ones_like(self.guesses[1][0][0])
                    start = 1
                else:
                    gd_last_idn = None
                    start = 0
            else:
                gd_last_idn = None
                start = 0
            # count all the combinations for discrete guesses
            discrete_counts = []
            for k,gd in enumerate(self.guesses[1]):
                if k<start:
                    continue
                discrete_counts.append(torch.arange(len(gd)))
            # enumerate all the discrete data guesses
            if len(discrete_counts) == 0:
                # when there is only one discrete data and it's identified by the last_grad
                self.guesses_init(continuous_shapes, discrete_shapes, discrete_range, disable_discrete_init=True)
                guess = self.guesses[0]
                if gd_last_idn is not None:
                    guess.append(gd_last_idn)
                if err_direc is not None:
                    guess.append(err_direc)
                self.attack_instance(gradient_label, gradient_generator, guess) 
                while self.nan_occur:
                    print('nan occur, run gradient inversion attack instance again')
                    self.nan_occur = False
                    self.guesses_init(continuous_shapes, discrete_shapes, discrete_range, disable_discrete_init=True)
                    guess = self.guesses[0]
                    if gd_last_idn is not None:
                        guess.append(gd_last_idn)
                    if err_direc is not None:
                        guess.append(err_direc)
                    self.attack_instance(gradient_label, gradient_generator, guess) 
            else:
                # enumerate over all the combinations
                discrete_combs = torch.cartesian_prod(*discrete_counts) # all possible combinations
                if len(discrete_combs.shape) == 1:
                    discrete_combs = discrete_combs[:,None]
                for discrete_idn in discrete_combs:
                    self.guesses_init(continuous_shapes, discrete_shapes, discrete_range, disable_discrete_init=True)
                    guess = self.guesses[0]
                    if gd_last_idn is not None:
                        guess.append(gd_last_idn)
                        guess.append(err_direc)
                    if len(self.guesses[1][start:])>0:
                        for gd, idn in zip(self.guesses[1][start:], discrete_idn):
                            guess.append(gd[idn])
                    self.attack_instance(gradient_label, gradient_generator, guess) 
                    while self.nan_occur:
                        print('nan occur, run gradient inversion attack instance again')
                        self.guesses_init(continuous_shapes, discrete_shapes, discrete_range, disable_discrete_init=True)
                        guess = self.guesses[0]
                        if gd_last_idn is not None:
                            guess.append(gd_last_idn)
                            guess.append(err_direc)
                        if len(self.guesses[1][start:])>0:
                            for gd, idn in zip(self.guesses[1][start:], discrete_idn):
                                guess.append(gd[idn])
                        self.attack_instance(gradient_label, gradient_generator, guess) 

        if self.records is not None:
            if len(self.records['loss']) > 0:
                #best_recon_idn = np.array([r['recon'] for r in self.records['loss']]).argmin()
                recon_losses = np.array([r['recon'] for r in self.records['loss']])
                zero_idn = (recon_losses==0).nonzero()[0]
                if len(zero_idn) > 0:
                    best_recon_idn = zero_idn[-1]
                else:
                    best_recon = np.abs(recon_losses).min()
                    print('ready to locate the min recon loss in attacker')
                    best_recon_idn = (recon_losses == best_recon).nonzero()[0][-1]
                self.best_recon_idn = best_recon_idn
                best_smooth_idn = np.array([r['smooth'] for r in self.records['loss']]).argmin()
                self.best_smooth_idn = best_smooth_idn
                best_recon_iter = self.records['iter'][best_recon_idn]
                self.best_recon_iter = best_recon_iter
                best_smooth_iter = self.records['iter'][best_smooth_idn]
                self.best_smooth_iter = best_smooth_iter
                if self.path and self.records:
                    if self.save_best:
                        best_recon_record = {}
                        best_smooth_record = {}
                        for k in self.records.keys():
                            if k == 'gradient_label':
                                best_recon_record[k] = self.records[k]
                                best_smooth_record[k] = self.records[k]
                            else:
                                best_recon_record[k] = self.records[k][best_recon_idn]
                                best_smooth_record[k] = self.records[k][best_smooth_idn]
                        torch.save(best_recon_record, self.path[0]+'_'+str(best_recon_iter)+'_best_recon_record.pt') 
                        torch.save(best_smooth_record, self.path[0]+'_'+str(best_smooth_iter)+'_best_smooth_record.pt') 
                    elif self.save_all:
                        torch.save(self.records, self.path[0]+'_'+str(self.path[1])+'_record.pt') 
                        # number indicates how many guess initializations have been tried and stored
                        self.path[1] += 1

        if self.img_path:
            if self.records is not None:
                best_recon_guess = self.records['input_guess'][best_recon_idn]
                self.save_img(guess=best_recon_guess, params=self.constraint_params, name='_best_recon_'+str(best_recon_iter)+'.jpg')
                best_smooth_guess = self.records['input_guess'][best_smooth_idn]
                self.save_img(guess=best_smooth_guess, params=self.constraint_params, name='_best_smooth_'+str(best_smooth_iter)+'.jpg')
            else:
                self.save_img(guess=self.best_guess, params=self.constraint_params, name='_best_guess.jpg')

    def attack_instance(self, gradient_label, gradient_generator, guess):
        self.iter = 0
        for k in range(self.max_iterations):
            self.optimizer.zero_grad()
            if gradient_generator is None:
                gradient_guess = self.gradient_generator(guess)
            else:
                gradient_guess = gradient_generator(self.shadow_model, guess, self.aux)
            loss, log_loss = self.loss_calculation(gradient_guess, gradient_label, guess)
            if log_loss['recon'] == 1:
                import pdb 
                pdb.set_trace()
            if torch.isnan(loss):
                warnings.warn('the final loss is nan, the individual losses are {}'.format(log_loss))
                self.nan_occur = True
                return guess, self.records
            loss.backward()
            if self.process_grad:
                self.modify_grad(guess)
            self.optimizer.step()
            self.scheduler.step()
            #print('lr ', self.optimizer.param_groups[0]['lr'] )
            self.prior_constraint(guess)
            if self.records:
                self.records['loss'].append(log_loss)
                self.records['input_guess'].append([guess.detach().cpu().numpy() for guess in guess])
                self.records['iter'].append(k)
            if np.abs(log_loss['recon']) <= self.best_recon_loss:
                self.best_guess = guess
                self.best_gradient_guess = gradient_guess 
                self.best_recon_loss = np.abs(log_loss['recon'])
                
            if k%self.print_freq == 0 or k == self.max_iterations-1:
                if self.records is None:
                    print(k, ' iters, last loss: ', log_loss)
                elif len(self.records['loss']) == 0:
                    print(k, ' iters')
                else:
                    print(k, ' iters, last loss: ', self.records['loss'][-1])
            
            if True: # this is an image saving paragraph for debugging
                if (k+1)%self.save_img_freq == 0:
                    if self.img_path:
                        self.save_img(params = self.constraint_params)
            
            if k==10000:
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
                if self.records is not None and self.save_loss:
                    last_record['all_loss'] = self.records['loss'][-self.iter:]
                torch.save(last_record, self.path[0]+'_'+str(self.path[1])+'_last_record.pt') 
            if self.save_all and self.records:
                torch.save(self.records, self.path[0]+'_'+str(self.path[1])+'_record.pt') 
            # number indicates how many guess initializations have been tried and stored
            self.path[1] += 1
        if self.img_path and (not k == 10000):
            self.save_img(params=self.constraint_params, name='_last')
            self.save_img(guess=self.best_guess, params=self.constraint_params, name='_best_guess')

        return guess, self.records


    def oracle(self, known):
        assert isinstance(known, list), 'the oracle information has to be passed in the form of list'
        assert len(known) == 2, 'the oracle information has to be a list of 2 elements, each of which is either a list or None'
        self.known_continuous = known[0]
        if self.known_continuous is not None:
            assert len(self.guesses[0]) == len(self.known_continuous), 'found {} continuous variables, but {} oracle messages'.format(len(self.guesses[0]), len(self.known_continuous))
        self.known_discrete = known[1]
        if self.known_discrete is not None:
            assert len(self.guesses[1]) == len(self.known_discrete),'found {} discrete variables, but {} oracle messages'.format(len(self.guesses[1]), len(self.known_discrete))


    def set_up_gradient_generator(self, model, inference, loss_fn):
        """
        this is for the most simple case, where there is only one input, only one output, and only one label, 
        and the loss function is provided with only 2 inputs, i.e. the output and label
        """
        if isinstance(model, list):
            self.shadow_model = [deepcopy(m) for m in model]
        else:
            self.shadow_model = deepcopy(model)
        self.shadow_loss = loss_fn
        self.shadow_inference = inference
    

    def gradient_generator(self, guesses):
        guess_input = []
        for g in guesses:
            if len(g.shape) == 4:
                if self.use_act:
                    g = self.act(g)
            guess_input.append(g)
        self.shadow_model.zero_grad()
        output = self.shadow_inference(self.shadow_model, guess_input)
        loss = self.shadow_loss(output, guesses)
        gradient_guess = torch.autograd.grad(loss, self.shadow_model.parameters(),
                            create_graph=True, allow_unused=True) #, retain_graph=True) 
        grad_guess = []
        for g in gradient_guess:
            if g is not None:
                grad_guess.append(g)
        return grad_guess


    def save_record(self, other=None):
        if self.path is not None and self.records is not None:
            if self.save_best:
                best_recon_record = {}
                best_smooth_record = {}
                for k in self.records.keys():
                    if k == 'gradient_label':
                        best_recon_record[k] = self.records[k]
                        best_smooth_record[k] = self.records[k]
                    else:
                        best_recon_record[k] = self.records[k][self.best_recon_idn]
                        best_smooth_record[k] = self.records[k][self.best_smooth_idn]
                if other is not None:
                    assert isinstance(other, dict)
                    for k in other.keys():
                        best_recon_record[k] = other[k]
                        best_smooth_record[k] = other[k]
                torch.save(best_recon_record, self.path[0]+'_'+str(self.best_recon_iter)+'_best_recon_record.pt') 
                torch.save(best_smooth_record, self.path[0]+'_'+str(self.best_smooth_iter)+'_best_smooth_record.pt') 
            elif self.save_all:
                torch.save(self.records, self.path[0]+'_'+str(self.path[1])+'_record.pt') 
                # number indicates how many guess initializations have been tried and stored
                self.path[1] += 1


    def _save_img(self, g, params, pcnt, name, mid_name):
        flag = False
        if len(g.shape) == 2:
            flag = True
        elif len(g.shape) == 3 and g.shape[0] == 3:  
            flag = True 
        elif len(g.shape) == 4:
            raise NotImplementedError('the tensor with shape {} is not supported, and current save_img doesn\'t support batch processing'.format(g.shape))
        if flag:
            if self.img_transpose:
                if len(g.shape) == 2:
                    g = g.transpose(0,1)
                elif len(g.shape) == 3:
                    g = g.transpose(1,2)
            if self.use_act:
                img = self.act(g)
            else:
                img = g.detach()
            if params is not None:
                param = params[pcnt]
                if len(param) == 2:
                    img = torch.clamp(img, min=param[0], max=param[1])
                elif len(param) == 4:
                    img = img * param[1] + param[0]
                    img = torch.clamp(img, min=param[2], max=param[3])
            torch.clamp(img, 0, 1)
            if name == '':
                path = self.img_path[0] + '_' + str(self.iter) + '_guess' + str(self.img_path[1]) + mid_name + '.jpg'
            else:
                if not name.endswith('.jpg'):
                    _name = name + mid_name + '.jpg'
                else:
                    _name = name.replace('.jpg', mid_name+'.jpg')
                path = self.img_path[0] + str(self.img_path[1]) + _name
            self.img_path[1] += 1
            torchvision.utils.save_image(img, path)
            print('save the image from the guess of continuous data: ', name, mid_name)


    def _save_img_sequence(self, g_seq, params, seq, name, mid_name):
        flag = False
        for g in g_seq:
            if len(g.shape) == 2:
                flag = True
            elif len(g.shape) == 3 and g.shape[0] == 3:  
                flag = True 
            elif len(g.shape) == 4:
                raise NotImplementedError('the tensor with shape {} is not supported'.format(g.shape))
        if flag:
            if self.img_transpose:
                for gcnt, g in enumerate(g_seq):
                    if len(g.shape) == 2:
                        g_seq[gcnt] = g.transpose(0,1)
                    elif len(g.shape) == 3:
                        g_seq[gcnt] = g.transpose(1,2)
            if self.use_act:
                for gcnt, g in enumerate(g_seq):
                    g_seq[gcnt] = self.act(g)
                else:
                    g_seq[gcnt] = g.detach()
            for (gcnt, g), s in zip(enumerate(g_seq), seq):
                if params[s] is not None:
                    param = params[s]
                    if len(param) == 2:
                        img = torch.clamp(g, min=param[0], max=param[1])
                    elif len(param) == 4:
                        img = g * param[1] + param[0]
                        img = torch.clamp(img, min=param[2], max=param[3])
                    torch.clamp(img, 0, 1)

            if len(g_seq[0].shape) == 3:
                g_seq = [np.uint8(g.detach().cpu().numpy()*256).transpose(1,2,0) for g in g_seq]
            elif len(g_seq[0].shape) == 2:
                g_seq = [np.uint8(g.detach().cpu().numpy()*256) for g in g_seq]
            if name == '':
                path = self.img_path[0] + '_' + str(self.iter) + '_guess' + str(self.img_path[1]) + mid_name + '.gif'
            else:
                if not name.endswith('.gif'):
                    _name = name + mid_name + '.gif'
                else:
                    _name = name.replace('.gif', mid_name+'.gif')
                path = self.img_path[0] + str(self.img_path[1]) + _name
            self.img_path[1] += 1
            import imageio
            imageio.mimsave(path, g_seq)


    def save_img(self, guess=None, params=None, name=''):
        if guess is None:
            guess = self.guesses[0]
        else:
            if torch.is_tensor(guess[0]):
                pass 
            else:
                guess = [torch.FloatTensor(g) for g in guess]
        if self.img_name_generator is None:
            mid_name = ''
        else:
            mid_name = self.img_name_generator(guess)

        if self.sequence_index is None:
            for pcnt, g in enumerate(guess):
                if pcnt == len(self.constraint_options):
                    break
                if self.constraint_options[pcnt] == 'image':
                    g = g.squeeze()
                    self._save_img(g=g, params=params, pcnt=pcnt, name=name, mid_name=mid_name)
        else:
            for seq in self.sequence_index:
                g = [guess[s].squeeze() for s in seq]
                self._save_img_sequence(g_seq=g, params=params, seq=seq, name=name, mid_name=mid_name)
                g = torch.cat(g, dim=-1)
                self._save_img(g=g, params=params, pcnt=seq[0], name=name, mid_name=mid_name)
                    
            for pcnt, g in enumerate(guess[0]):
                existed = False
                for seq in self.sequence_index:
                    if pcnt in seq:
                        existed = True
                if existed:
                    continue
                
                if pcnt == len(self.constraint_options):
                    break
                if self.constraint_options[pcnt] == 'image':
                    self._save_img(g=g, params=params, pcnt=pcnt, name=name, mid_name=mid_name)


    def save_img_gt(self, img, param=None, name='', path=''):
        if path == '':
            if self.img_path is not None:
                path = self.img_path[0]
            else:
                path = './'
        if isinstance(img, torch.Tensor):
            img = img.squeeze()
            if self.img_transpose:
                img = img.transpose(-1,-2)
        elif isinstance(img, list):
            imgs = []
            for im in img:
                im.squeeze()
                if self.img_transpose:
                    im = im.transpose(-1,-2)
                imgs.append(im)
            img = torch.cat(imgs, dim=-1)
        if param is not None:
            if len(param) == 2: 
                img = torch.clamp(img, low=param[0], up=param[1])
            elif len(param) == 4:
                img = img * params[1] + params[0]
                img = torch.clamp(img, low=param[2], up=param[3])
        if name == '':
            name = 'gt'
        torchvision.utils.save_image(img, path + name + '.jpg')
        print('save image ', name)


    def save_img_sequence_gt(self, img_seq, params=None, name='', path=''):
        if path == '':
            if self.img_path is not None:
                path = self.img_path[0]
            else:
                path = './'
        assert isinstance(img_seq, list), 'the input of save_img_sequence_gt has to be a list'
        imgs = []
        for im in img_seq:
            im.squeeze()
            if self.img_transpose:
                im = im.transpose(-1,-2)
            imgs.append(im)
        if params is not None:
            for (cnt,img), param in zip(imgs, params):
                if len(param) == 2: 
                    img = torch.clamp(img, low=param[0], up=param[1])
                elif len(param) == 4:
                    img = img * params[1] + params[0]
                    img = torch.clamp(img, low=param[2], up=param[3])
                imgs[cnt] = img
        if len(imgs[0].shape) == 3:
            imgs = [np.uint8(img.cpu().numpy()*256).transpose(1,2,0) for img in imgs]
        elif len(imgs[0].shape) == 2:
            imgs = [np.uint8(img.cpu().numpy()*256) for img in imgs]
        else:
            raise RuntimeError('current save_img_sequence_gt only support 3-channel and 1-channel image')
        if name == '':
            name = 'gt'
        import imageio
        imageio.mimsave(path + name + '_sequence.gif', imgs)
        print('save image ', name)
