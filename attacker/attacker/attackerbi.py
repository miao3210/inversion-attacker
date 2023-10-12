from attacker.attacker_base import GradientAttacker
import warnings
import torch 
import torch.nn as nn 

class GradientAttackerBiPart(GradientAttacker):
    def __init__(self,
        lrs=[0.1, 0.1, 0.01],
        max_iterations=[10000, 10, 25000],
        continuous_index=[[],[],[]],
        gradient_index=[[],[],[]],
        **args):
        super().__init__(**args)

        self.lr = None
        self.lrs = lrs
        self.max_iterations = max_iterations
        self.continuous_index = continuous_index
        self.gradient_index = gradient_index
        assert isinstance(self.max_iterations, list) or isinstance(self.max_iterations, tuple), 'the max_iterations of attack with breakpoint has to be a list or tuple with 3 elements'
        assert len(self.continuous_index) == 3, 'current attacker has 3 phases, so the continuous index has to have 3 elements'
        
        self.feat_coeff = 0.1

    def feat_loss(self, feat_conv, feat_fc):
        loss = nn.MSELoss()
        return loss(feat_conv, feat_fc)
    
    def loss_calculation(self, gradient_guess, gradient_label, guess, feat, phase):
        grad_recon = self.recon_loss(gradient_guess, gradient_label) 
        img_smoothness = self.smooth_coeff * self.smoothness(guess)
        log = {
            'recon': grad_recon.item(),
            'smooth': img_smoothness.item(),
        }
        if phase in ['fc']:
            loss = grad_recon #+ img_smoothness
        elif phase in ['conv']:
            feat_brk = self.feat_coeff * self.feat_loss(feat, self.guesses[2][0])
            loss = feat_brk # += feat_brk
            log['breakpoint matching'] = feat_brk.item()
        elif phase in ['conv', 'both']:
            feat_brk = self.feat_coeff * self.feat_loss(feat, self.guesses[2][0])
            loss = grad_recon + img_smoothness + feat_brk
            log['breakpoint matching'] = feat_brk.item()
        return loss, log


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


    def attack_instance(self, gradient_label, gradient_generator, guess, max_iteration, constraint_options, constraint_params, phase):
        for k in range(max_iteration):
            if k%1000 == 1:
                if self.records is None:
                    print(k-1, ' iters, last loss: ', log_loss)
                #elif len(self.records['loss']) == 0:
                #    print(k-1, ' iters')
                else:
                    print(k-1, ' iters, last loss: ', self.records['loss'][-1])
            if k%5000 == 0:
                if self.img_path:
                    self.save_img(params = self.constraint_params)
            
            self.optimizer.zero_grad()
            if gradient_generator is None:
                gradient_guess, feat = self.gradient_generator(guess, phase=phase)
            else:
                gradient_guess, feat = gradient_generator(guess, phase=phase)
            loss, log_loss = self.loss_calculation(gradient_guess, gradient_label, guess, feat, phase=phase)
            if torch.isnan(loss):
                import pdb 
                pdb.set_trace()
            loss.backward()
            if self.signed_grad:
                self.modify_grad(guess)
            self.optimizer.step()
            self.scheduler.step()
            #print('lr ', self.optimizer.param_groups[0]['lr'] )
            self.prior_constraint(guess, constraint_options, constraint_params)
            if self.records:
                self.records['loss'].append(log_loss)
                self.records['input_guess'].append([guess.detach().cpu().numpy() for guess in guess])

        if phase == 'fc':
            feature_brkpoint = guess[0].detach()
            feature_brkpoint.requires_grad = True
            output, _ = self.shadow_inference(self.shadow_model, [feature_brkpoint]+guess[1:], phase=phase)
            loss = self.shadow_loss(output, guess)
            derivative_brkpoint = torch.autograd.grad(loss, feature_brkpoint)
            self.guesses.append([feature_brkpoint.detach(), derivative_brkpoint[0].detach()])
        if self.path:
            sio.savemat(self.path[0]+'_'+str(self.path[1])+'.mat', self.records) 
            # number indicates how many guess initializations have been tried and stored
            self.path[1] += 1
        if self.img_path:
            self.save_img(params = self.constraint_params)
        return guess, self.records


    def attack(self, gradient_label, gradient_generator=None, bsz1=False, case=None, last_idn=None, 
                    continuous_shapes=None, discrete_shapes=None, discrete_range=None, continuous_index=None):
        if continuous_shapes:
            self.continuous_shapes = continuous_shapes
        if discrete_shapes:
            self.discrete_shapes = discrete_shapes
        if discrete_range:
            self.discrete_range = discrete_range
        assert len(self.discrete_shapes) == len(self.discrete_range)
        if continuous_index:
            self.continuous_index = continuous_index
        
        if len(self.guesses[1]) == 0:
            # when all guess_data are continuous
            for phase, max_iteration, index, lr, gradindex in zip(['fc','conv','both'], self.max_iterations, self.continuous_index, self.lrs, self.gradient_index):
                if phase == 'both':
                    prior_initialization = [self.guesses[0][0], self.guesses[0][1], None]
                else:
                    prior_initialization = None
                self.guesses_init(self.continuous_shapes[index], self.discrete_shapes, self.discrete_range, lr, max_iteration, self.known_continuous[index], prior_initialization=prior_initialization)
                guess = self.guesses[0]
                constraint_options = self.constraint_options[index]
                constraint_params = self.constraint_params[index]
                self.attack_instance(gradient_label[gradindex], gradient_generator, guess, max_iteration, constraint_options, constraint_params, phase) 
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
                elif case == 'rl':
                    assert len(last_grad[last_grad!=0]) == 1
                    gd_last_idn = torch.nonzero(last_grad).view(-1) * torch.ones_like(self.guesses[1][0][0])
                else:
                    gd_last_idn = None
                start = 1
            else:
                start = 0
            # count all the combinations for discrete guesses
            discrete_counts = []
            for k,gd in enumerate(self.guesses[1]):
                if k < start:
                    continue
                discrete_counts.append(torch.arange(len(gd)))
            # enumerate all the discrete data guesses
            if len(discrete_counts) == 0:
                # when there is only one discrete data and it's identified by the last_grad
                for phase, max_iteration, index, lr, gradindex in zip(['fc','conv','both'], self.max_iterations, self.continuous_index, self.lrs, self.gradient_index):
                    if phase == 'both':
                        prior_initialization = [self.guesses[0][0], self.guesses[0][1], None]
                    else:
                        prior_initialization = None
                    self.guesses_init(self.continuous_shapes[index], self.discrete_shapes, self.discrete_range, disable_discrete_init=True, lr=lr, max_iterations=max_iteration, known_continuous=self.known_continuous[index], prior_initialization=prior_initialization)
                    guess = self.guesses[0]
                    if gd_last_idn is not None:
                        guess.append(gd_last_idn)
                    constraint_options = self.constraint_options[index]
                    constraint_params = self.constraint_params[index]
                    self.attack_instance(gradient_label[gradindex], gradient_generator, guess, max_iteration, constraint_options, constraint_params, phase) 
            else:
                # enumerate over all the combinations
                discrete_combs = torch.cartesian_prod(*discrete_counts) # all possible combinations
                if len(discrete_combs.shape) == 1:
                    discrete_combs = discrete_combs[None]
                for discrete_idn in discrete_combs:
                    for phase, max_iteration, index, lr, gradindex in zip(['fc','conv','both'], self.max_iterations, self.continuous_index, self.lrs, self.gradient_index):
                        if phase == 'both':
                            prior_initialization = [self.guesses[0][0], self.guesses[0][1], None]
                        else:
                            prior_initialization = None
                        self.guesses_init(self.continuous_shapes[index], self.discrete_shapes, self.discrete_range, disable_discrete_init=True, lr=lr, max_iterations=max_iteration, known_continuous=self.known_continuous[index], prior_initialization=prior_initialization)
                        guess = self.guesses[0]
                        if gd_last_idn is not None:
                            guess.append(gd_last_idn)
                        for gd, idn in zip(self.guesses[1][start:], discrete_idn):
                            guess.append(gd[idn])
                        constraint_options = self.constraint_options[index]
                        constraint_params = self.constraint_params[index]
                        self.attack_instance(gradient_label[gradindex], gradient_generator, guess, max_iteration, constraint_options, constraint_params, phase) 
                    
