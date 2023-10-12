import torch 
import torch.nn as nn
import warnings

class Loss():
    def __init__(self, gradient_active,
                    loss_type,
                    use_act,
                    device='cuda',
                    constraint_options=None):
        self.gradient_active = gradient_active 
        self.loss_type = loss_type
        self.use_act = use_act
        self.act = None
        self.device = device
        self.constraint_options = constraint_options
        self.waive = 0
        self.sequence_index = None


    def recon_loss(self, gradient_guess, gradient_label):
        if self.gradient_active is not None:
            _gradient_label = []
            _gradient_guess = []
            if self.gradient_active[0] is not None:
                for slic in self.gradient_active[0]:
                    if isinstance(gradient_label[slic], list):
                        _gradient_label += gradient_label[slic]
                    else:
                        _gradient_label.append(gradient_label[slic])
            else:
                _gradient_label = gradient_label
            if self.gradient_active[1] is not None:
                for slic in self.gradient_active[1]:
                    if isinstance(gradient_guess[slic], list):
                        _gradient_guess += gradient_guess[slic]
                    else:
                        _gradient_guess.append(gradient_guess[slic])
            else:
                _gradient_guess = gradient_guess
        else:
            _gradient_guess = gradient_guess
            _gradient_label = gradient_label         
        
        if self.loss_type == 'sim':
            return self.sim_loss(_gradient_guess, _gradient_label)
        elif self.loss_type == 'layerwise_sim':
            return self.layerwise_sim_loss(_gradient_guess, _gradient_label)
        else:
            if self.loss_type == 'l1':
                loss = nn.L1Loss().to(self.device)
            elif self.loss_type == 'l2' or 'mse':
                loss = nn.MSELoss().to(self.device)
            else:
                raise RuntimeError('current attacker only support 3 reconstruction loss type: l1, l2, sim, but got {}}'.format(self.loss_type))
            recon_loss = 0
            for gg, gl in zip(_gradient_guess, _gradient_label):
                recon_loss += loss(gg, gl)
            return recon_loss 


    def sim_loss(self, gradient_guess, gradient_label):
        '''
        cosine similarity loss from https://arxiv.org/abs/2003.14053
        '''
        sim_loss = 0
        gg_norm = 0
        gl_norm = 0
        for gg, gl in zip(gradient_guess, gradient_label):
            sim_loss -= (gg * gl).sum()
            #with torch.no_grad():
            if True:
                gg_norm += gg.pow(2).sum()
                gl_norm += gl.pow(2).sum()
        if gg_norm > 0 and gl_norm > 0:
            sim_loss = 1 + sim_loss / gg_norm.sqrt() / gl_norm.sqrt()
        return sim_loss
    

    def layerwise_sim_loss(self, gradient_guess, gradient_label):
        warnings.warn('layerwise_sim_loss assumes that each layer has 2 parameters to optimize')
        cnt_params = len(gradient_label)
        assert cnt_params > 0 and cnt_params % 2 == 0, 'the layerwise sim_loss assume all layers have 2 parameters'
        sim_loss = 0
        cnt = 0
        prod = 0
        gg_norm = 0 
        gl_norm = 0
        for gg, gl in zip(gradient_guess, gradient_label):
            prod += (gg * gl).sum()
            gg_norm += gg.pow(2).sum()
            gl_norm += gl.pow(2).sum()
            cnt += 1
            if cnt % 2 == 0:
                layer_sim_loss = 1 - prod / gg_norm.sqrt() / gl_norm.sqrt()
                sim_loss += layer_sim_loss
                prod = 0 
                gg_norm = 0
                gl_norm = 0
        return sim_loss / (cnt_params / 2)


    def total_variation(self, x):
        """Anisotropic TV.
        this is from https://github.com/JonasGeiping/invertinggradients"""
        warnings.warn('tv loss assumes the shape of the image is (B,C,H,W)')
        if self.use_act:
            x = self.act(x)
        #dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        #dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        #return dx + dy
        dx = x[:, :, :, :-1] - x[:, :, :, 1:]
        dy = x[:, :, :-1, :] - x[:, :, 1:, :]
        loss = 0
        for d in [dx, dy]:
            dabs = d.abs()
            if self.waive > 0:
                idn = dabs.view(-1).argsort()[:-self.waive]
                loss += torch.mean(dabs.view(-1)[idn])
            else:
                loss += torch.mean(dabs)
        return loss


    def total_variation_high_order(self, x):
        warnings.warn('tv-based losses assumes the shape of the image is (B,C,H,W)')
        if self.use_act:
            x = self.act(x)
        x = nn.functional.pad(x, (2,2,2,2), 'reflect')
        dx = x[:, :, :, :-1] - x[:, :, :, 1:]
        dy = x[:, :, :-1, :] - x[:, :, 1:, :]
        dxdx = dx[:, :, :, :-1] - dx[:, :, :, 1:]
        dxdy = dx[:, :, :-1, :] - dx[:, :, 1:, :]
        dydx = dy[:, :, :, :-1] - dy[:, :, :, 1:]
        dydy = dy[:, :, :-1, :] - dy[:, :, 1:, :]
        loss = 0
        for d in [dxdx, dxdy, dydx, dydy]:
            dabs = d.abs()
            if self.waive > 0:
                idn = dabs.view(-1).argsort()[:-self.waive]
                loss += torch.mean(dabs.view(-1)[idn])
            else:
                loss += torch.mean(dabs)
        return loss


    def smoothness(self, guess, order=1):
        warnings.warn('image prior loss select input guess with heuristic rules')
        smooth = 0
        if self.constraint_options is not None:
            for k,g in enumerate(guess):
                if k < len(self.constraint_options):
                    if self.constraint_options[k] == 'image':
                        if len(g.shape) == 4 and g.requires_grad is True:
                            if order == 1:
                                smooth += self.total_variation(g) #_high_order(g) / g.detach().abs().mean() #+ g.std()
                            elif order == 2:
                                smooth += self.total_variation_high_order(g)
                            else:
                                raise NotImplementedError('current smoothness loss only support 1st-order and 2nd-order tv loss')
        return smooth

    def smoothness_independent(self, guess, order=1):
        warnings.warn('image prior loss select input guess with heuristic rules')
        smooth = []
        if self.constraint_options is not None:
            for k,g in enumerate(guess):
                if k < len(self.constraint_options):
                    if self.constraint_options[k] == 'image':
                        if len(g.shape) == 4 and g.requires_grad is True:
                            if order == 1:
                                smooth.append( self.total_variation(g) ) #_high_order(g) / g.detach().abs().mean() #+ g.std()
                            elif order == 2:
                                smooth.append( self.total_variation_high_order(g) )
                            else:
                                raise NotImplementedError('current smoothness loss only support 1st-order and 2nd-order tv loss')
        return torch.tensor(smooth, device=smooth[0].device)


    def brightness(self, guess):
        warnings.warn('image prior loss select input guess with heuristic rules')
        bright = 0
        if self.constraint_options is not None:
            for k,g in enumerate(guess):
                if k < len(self.constraint_options):
                    if self.constraint_options[k] == 'image':
                        if len(g.shape) == 4 and g.requires_grad is True:
                            bright -= (g.abs()).mean()
        return bright


    def brightness_independent(self, guess):
        warnings.warn('image prior loss select input guess with heuristic rules')
        bright = []
        if self.constraint_options is not None:
            for k,g in enumerate(guess):
                if k < len(self.constraint_options):
                    if self.constraint_options[k] == 'image':
                        if len(g.shape) == 4 and g.requires_grad is True:
                            bright.append( - (g.abs()).mean())
        return torch.tensor(bright, device=bright[0].device)


    def set_sequence_loss(self, sequence_index):
        assert isinstance(sequence_index, list) or isinstance(sequence_index, tuple)
        for idn in sequence_index:
            assert isinstance(idn, tuple) or isinstance(idn, list), 'the sequence index has to be either list of list or list of tuple, but got {}'.format(sequence_index)
            for sample_idn in idn:
                assert isinstance(sample_idn, int), 'the sample index in the sequence index has to be integer, but got {}'.format(sample_idn)
        self.sequence_index = sequence_index
            

    def sequence_smoothness(self, guess):
        sequential_smooth = 0
        if self.sequence_index is not None:
            for seq in self.sequence_index:
                seq_loss = 0
                for t0,t1 in zip(seq[:-1], seq[1:]):
                    former = guess[t0]
                    latter = guess[t1]
                    #seq_loss += (latter - former).abs().mean()
                    seq_loss += (latter.mean() - former.mean()).abs() #+ 0.1 * (latter - former).abs().mean()
                sequential_smooth += seq_loss #.mean()
        return sequential_smooth


    def sequence_smoothness_independent(self, guess):
        sequential_smooth = []
        if self.sequence_index is not None:
            for seq in self.sequence_index:
                seq_loss = []
                for t0,t1 in zip(seq[:-1], seq[1:]):
                    former = guess[t0]
                    latter = guess[t1]
                    #seq_loss += (latter - former).abs().mean()
                    seq_loss += (latter.mean() - former.mean()).abs() #+ 0.1 * (latter - former).abs().mean()
                sequential_smooth.append( seq_loss ) #.mean()
        return torch.tensor(sequential_smooth, device=sequential_smooth[0].device)
