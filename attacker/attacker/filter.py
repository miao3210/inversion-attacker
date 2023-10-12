import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import torchvision

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    author: https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=1, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class Mask(nn.Module):
    """
    This filter is designed for binary mask with retangle region of 1, while the background is 0
    """
    def __init__(self, win=2, thre=3):
        super().__init__()
        self.win = win
        self.thre = thre

    def forward(self, x):
        #x[x>0.8] = 1
        #x[x<0.8] = 1
        for k in range(8):
            x = self.rule(x)
            torchvision.utils.save_image(x, '/home/miao/APDP/active_perception-defense/images_dqn/rule_'+str(k)+'.jpg')
        
        import pdb 
        pdb.set_trace()
        return self.rule(x)
        # return self.sum_thre(x)

    def rule(self, x):
        if len(x.shape) == 4:
            for k in range(x.shape[0]):
                for n in range(x.shape[1]):
                    mask = x[k,n]
                    for h in range(mask.shape[0]):
                        for w in range(mask.shape[1]):
                            mask.data[h,w] = self._rule_pixel(mask, h, w)
                    nonz = torch.nonzero(mask)
                    hmin = nonz[:,-2].min()
                    hmax = nonz[:,-2].max()
                    wmin = nonz[:,-1].min()
                    wmax = nonz[:,-1].max()
                    mask[hmin:(hmax+1), wmin:(wmax)] = torch.ones_like(mask[hmin:(hmax+1), wmin:(wmax)])
                    x.data[k,n] = mask.data

        elif len(x.shape) == 2:
            mask = x
            for h in range(mask.shape[0]):
                for w in range(mask.shape[1]):
                    mask.data[h,w] = self._rule_pixel(mask, h, w)
            nonz = torch.nonzero(mask)
            hmin = nonz[:,-2].min()
            hmax = nonz[:,-2].max()
            wmin = nonz[:,-1].min()
            wmax = nonz[:,-1].max()
            mask[hmin:(hmax+1), wmin:(wmax)] = torch.ones_like(mask[hmin:(hmax+1), wmin:(wmax)])
            x.data = mask.data
        return x

    def _rule_pixel(self, mask, h, w):
        flag = 0
        upper_edge = False
        lower_edge = False 
        left_edge = False 
        right_edge = False
        # above
        if h >= 2:
            above = mask[(h-2):h, w]
            if above[0] > 0.8 and above[1] > 0.8:
                flag += 1
        elif h == 1:
            above = mask[:h, w]
            if above[0] > 0.8:
                flag += 1
        else:
            upper_edge = True
        # below
        if h < mask.shape[0]-2:
            below = mask[h:(h+2), w]
            if below[0] > 0.8 and below[1] > 0.8:
                flag += 1
        elif h == mask.shape[0]-2:
            below = mask[h:, w]
            if below[0] > 0.8:
                flag += 1
        else:
            lower_edge == True 
        # left
        if w >= 2:
            left = mask[h, (w-2):w]
            if left[0] > 0.8 and left[1] > 0.8:
                flag += 1
        elif w == 1:
            left = mask[h, :w]
            if left[0] > 0.8:
                flag += 1
        else:
            left_edge = True
        # right
        if w < mask.shape[1]-2:
            right = mask[h, w:(w+2)]
            if right[0] > 0.8 and right[1] > 0.8:
                flag += 1 
        elif w == mask.shape[1]-2:
            right = mask[h, w:]
            if right[0] > 0.8:
                flag += 1
        else:
            right_edge = True
        ## summary
        if flag >= 2:
            return 1
        else: 
            return 0


    def sum_thre(self, x):
        if len(x.shape) == 4:
            for k in range(x.shape[0]):
                for n in range(x.shape[1]):
                    mask = x[k,n]
                    for h in range(mask.shape[0]-self.win+1):
                        for w in range(mask.shape[1]-self.win+1):
                            square = mask[h:(h+2), w:(w+2)]
                            if square.sum() > self.thre:
                                mask[h:(h+2), w:(w+2)] = torch.ones_like(mask[h:(h+2), w:(w+2)])
                    x.data[k,n] = mask.data
        elif len(x.shape) == 2:
            mask = x
            for h in range(mask.shape[0]-self.win+1):
                for w in range(mask.shape[1]-self.win+1):
                    square = mask[h:(h+2), w:(w+2)]
                    if square.sum() > self.thre:
                        mask[h:(h+2), w:(w+2)] = torch.ones_like(mask[h:(h+2), w:(w+2)])
            x.data = mask.data
        return x