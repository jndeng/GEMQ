import torch
import torch.nn as nn


class RTNWeightQuantizer(nn.Module):
    def __init__(self, x, name="", nbits=4, groupsize=-1):
        super().__init__()

        self.name = name
        self.nbits = nbits
        self.groupsize = groupsize

        self.W = x.clone()

    def quantize(self, W=None):
        """
        Round-to-nearest (RTN) pseudo quantization with min-max clip range (adapted from AWQ).
        """
        nbits, groupsize = self.nbits, self.groupsize
        if W is None:
            assert self.W is not None, "No weight tensor to quantize! Please provide W or initialize self.W"
            W = self.W

        org_w_shape = W.shape
        if groupsize > 0:
            assert org_w_shape[-1] % groupsize == 0, f"w_shape: {org_w_shape[-1]} | groupsize: {groupsize}"
            W = W.reshape(-1, groupsize) # [num_groups, groupsize]

        if nbits == 0:
            num_groups = W.shape[0]
            Q = torch.zeros_like(W)
            scales = torch.ones(num_groups, 1, dtype=W.dtype, device=W.device)
            zeros = torch.zeros(num_groups, 1, dtype=W.dtype, device=W.device)
            return Q, scales, zeros
    
        if nbits == 1:
            Q = torch.where(W >= 0, 1, 0).to(W.dtype)
            scales = torch.mean(torch.abs(W), dim=1, keepdim=True) * 2
            zeros = 0.5 * torch.ones_like(scales)
            return Q, scales, zeros

        max_val = W.amax(dim=1, keepdim=True)
        min_val = W.amin(dim=1, keepdim=True)

        max_int = 2 ** nbits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(W).sum() == 0

        Q = torch.clamp(torch.round(W / scales) + zeros, 0, max_int)

        return Q, scales, zeros
    
    def dequantize(self, Q, scales, zeros):
        """
        Dequantize a quantized tensor using the given quantization parameters.
        """
        return (Q - zeros) * scales
    
    def pseudo_quantize(self, W=None):
        """
        Quantize and dequantize the weight tensor.
        """
        W_q = self.dequantize(*self.quantize(W))
        if self.groupsize > 0:
            W_q = W_q.reshape(self.W.shape)
        return W_q


class MCMoeRTNWeightQuantizer(nn.Module):
    def __init__(self, x, nbits, blocksize=128):
        super().__init__()
        
        self.x = x.clone()
        self.nbits = nbits
        self.blocksize = blocksize

    @classmethod
    def binary(cls, x):
        scale_tensor = torch.mean(torch.abs(x), dim=1) * 2
        zero = 0.5 * torch.ones_like(scale_tensor)
        x_ = torch.zeros_like(x)
        x_ += x
        binary_slice = torch.where(x_ >= 0, 1, -1)
        binary_slice = binary_slice/2 + 0.5

        scale_tensor = scale_tensor.unsqueeze(1).expand(-1, 128)
        zero = zero.unsqueeze(1).expand(-1, 128)
        return scale_tensor * (binary_slice - zero)

    @classmethod
    def __quantize(cls, x, scale, zero, maxq):
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @classmethod
    def _quantize(cls, w, wbit):
        perchannel = True
        weight = True
        dev = w.device
        maxq = torch.tensor(2 ** wbit - 1)
        scale = torch.zeros(1)
        zero = torch.zeros(1)
        if dev != scale.device:
            scale=scale.to(dev)
            zero=zero.to(dev)
            maxq=maxq.to(dev)

        x = w.clone()
        shape = x.shape

        if perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        if not perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            scale = scale.reshape(shape)
            zero = zero.reshape(shape)
        w = cls.__quantize(w, scale, zero, maxq)

        return w

    @classmethod
    def normal_quantize(cls, w, blocksize=128, wbit=2):
        columns = w.shape[1]
        w_q = torch.zeros_like(w)
        w_q = w_q.to(w.device)
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1

            W1 = w[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            if wbit == 1:
                Q1 = cls.binary(W1)
            else:
                Q1 = cls._quantize(W1, wbit)
            
            w_q[:, i1:i2] = Q1
        return w_q

    def quantize(self):
        if self.nbits == 0:
            return torch.zeros_like(self.x)
        return MCMoeRTNWeightQuantizer.normal_quantize(self.x, self.blocksize, self.nbits)
