import math

import torch
import torch.nn as nn


class MCMoeGPTQWeightQuantizer(nn.Module):
    def __init__(
        self, x, name="", nbits=4, blocksize=128, percdamp=0.01,
        groupsize=-1, actorder=False, static_groups=False, mse=False,
    ):
        super().__init__()

        self.name = name
        self.nbits = nbits
        self.groupsize = groupsize
        assert self.groupsize > 0, "Current implementation only supports groupsize > 0."

        self.blocksize = blocksize
        self.percdamp = percdamp
        self.actorder = actorder
        self.static_groups = static_groups
        self.mse = mse

        # init
        self.rows = x.shape[0]     # N_filters
        self.columns = x.shape[1]  # N_feat_dim
        self.W = x.clone()
        self.H = torch.zeros((self.columns, self.columns), device=x.device)
        self.nsamples = 0

        assert self.columns % self.groupsize == 0, f"Groupsize ({self.groupsize}) must divide feature dimension ({self.columns})."

    def add_batch(self, inp):
        """
        Update Hessian matrix with the input tensor (a batch of sequences).
        """
        # inp (attn): (bsz, num_seq, num_dim)
        # inp (fc):   (bsz*num_seq, num_dim)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        bsz = inp.shape[0]

        inp = inp.reshape(-1, inp.shape[-1])  # (bsz*num_seq, num_dim)
        inp = inp.t()  # (num_dim, bsz*num_seq)

        self.H *= self.nsamples / (self.nsamples + bsz)
        self.nsamples += bsz
        
        inp = math.sqrt(2. / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())  # (num_dim, num_dim)

    def find_params(self, x):
        """
        Find quantization parameters (scales zero-points, and max_int) for a tensor.
        """
        x = x.to(self.W.dtype)

        max_int = 2 ** self.nbits - 1
        if max_int == 1:
            scales = torch.mean(torch.abs(x), dim=1, keepdim=True) * 2
            zeros = 0.5 * torch.ones_like(scales)
            return scales, zeros, max_int

        tmp = torch.zeros(x.shape[0], 1, dtype=self.W.dtype, device=x.device)  # (N_filter,1)
        min_val = torch.minimum(x.amin(dim=1, keepdim=True), tmp)  # (N_filter,1)
        max_val = torch.maximum(x.amax(dim=1, keepdim=True), tmp)  # (N_filter,1)
        
        tmp = (min_val == 0) & (max_val == 0)
        min_val[tmp] = -1
        max_val[tmp] = +1

        scales = (max_val - min_val) / max_int  # (N_filter,1)
        zeros = -min_val / scales  # (N_filter,1)

        if self.mse:
            norm = 2.4

            tau_range = 0.1
            tau_n = 50
            p_left  = 1 - tau_range
            p_right = 1 + tau_range

            best = torch.full([x.shape[0]], float("inf"), device=x.device, dtype=self.W.dtype)  # (N_filters,)
            for _, p in enumerate(torch.cat([torch.ones(1), torch.linspace(1.0, p_right, tau_n+1)[1:], torch.linspace(1.0, p_left, tau_n+1)[1:]])):
                minv = p * min_val
                maxv = p * max_val

                tmp_scales = (maxv - minv) / max_int
                tmp_zeros = -minv / tmp_scales

                q = self.quantize_vector(x, tmp_scales, tmp_zeros, max_int)
                q = tmp_scales * (q - tmp_zeros)

                q -= x
                q.abs_()
                q.pow_(norm)
                err = torch.sum(q, 1)

                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    scales[tmp] = tmp_scales[tmp]
                    zeros[tmp] = tmp_zeros[tmp]

        return scales, zeros, max_int

    @staticmethod
    def quantize_vector(x, scales, zeros, max_int):
        """
        Quantize a tensor using the given quantization parameteres.
        """
        # quantize
        if max_int == 1:
            x = torch.where(x >= 0, 1, -1)
            x = x / 2 + 0.5
        else:
            x = torch.clamp(torch.round(x / scales + zeros), 0, max_int)

        return x

    def quantize(self):
        """
        Perform GPTQ quantization for the weight tensor.
        """
        device = self.W.device

        if self.nbits == 0:
            num_groups = self.W.numel() // self.groupsize if self.groupsize > 0 else self.W.shape[0]
            Q = torch.zeros_like(self.W).reshape(num_groups, -1)
            scales = torch.ones(num_groups, 1, dtype=self.W.dtype, device=device)
            zeros = torch.zeros(num_groups, 1, dtype=self.W.dtype, device=device)
            return Q, scales, zeros
        

        W = self.W.float()  # (n_fltd, n_dim)
        H = self.H          # (n_dim, n_dim)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        assert not self.static_groups
        if self.static_groups:
            pass

        assert not self.actorder
        if self.actorder:
            pass

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # initialize quantization params (assuming groupsize < 0)
        scales, zeros, max_int = self.find_params(W)
        scales_list = []
        zeros_list = []

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.groupsize > 0:
                    if not self.static_groups:
                        # update quantization params for a new group
                        if (i1 + i) % self.groupsize == 0:
                            scales, zeros, max_int = self.find_params(W[:, (i1 + i):(i1 + i + self.groupsize)])
                            scales_list.append(scales)
                            zeros_list.append(zeros)
                    else:
                        pass

                q = self.quantize_vector(w.unsqueeze(1), scales, zeros, max_int)
                qr = (q - zeros) * scales  # dequantize

                q = q.flatten()
                qr = qr.flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - qr) ** 2 / d ** 2
                err1 = (w - qr) / d
                
                if self.nbits > 3:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

                Err1[:, i] = err1
            
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.actorder:
            pass

        Q = Q.to(self.W.dtype)
        scales = torch.cat(scales_list, dim=1)  # (N_filter, N_group)
        zeros = torch.cat(zeros_list, dim=1)  # (N_filter, N_group)

        # reformat
        Q = Q.reshape(-1, self.groupsize) # (N_group, groupsize)
        scales = scales.reshape(-1, 1)    # (N_group, 1)
        zeros = zeros.reshape(-1, 1)      # (N_group, 1)

        return Q, scales, zeros

    def dequantize(self, Q, scales, zeros):
        """
        Dequantize a quantized tensor using the given quantization parameters.
        """
        return (Q - zeros) * scales


class GPTQWeightQuantizer(nn.Module):
    def __init__(
        self, x, name="", nbits=4, blocksize=128, percdamp=0.01,
        groupsize=-1, actorder=False, static_groups=False, mse=False,
    ):
        super().__init__()

        self.name = name
        self.nbits = nbits
        self.groupsize = groupsize

        self.blocksize = blocksize
        self.percdamp = percdamp
        self.actorder = actorder
        self.static_groups = static_groups
        self.mse = mse

        # init
        self.rows = x.shape[0]     # N_filters
        self.columns = x.shape[1]  # N_feat_dim
        self.W = x.clone()
        self.H = torch.zeros((self.columns, self.columns), device=x.device)
        self.nsamples = 0

    def add_batch(self, inp):
        """
        Update Hessian matrix with the input tensor (a batch of sequences).
        """
        # inp (attn): (bsz, num_seq, num_dim)
        # inp (fc):   (bsz*num_seq, num_dim)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        bsz = inp.shape[0]

        inp = inp.reshape(-1, inp.shape[-1])  # (bsz*num_seq, num_dim)
        inp = inp.t()  # (num_dim, bsz*num_seq)

        if torch.isnan(inp).any():
            raise ValueError(self.nsamples, "NaN detected in inp.")

        self.H *= self.nsamples / (self.nsamples + bsz)
        self.nsamples += bsz

        inp = math.sqrt(2. / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())  # (num_dim, num_dim)

    def find_params(self, x):
        """
        Find quantization parameters (scales zero-points, and max_int) for a tensor.
        """
        max_int = 2 ** self.nbits - 1
        if max_int == 1:
            scales = torch.mean(torch.abs(x), dim=1, keepdim=True) * 2
            zeros = 0.5 * torch.ones_like(scales)
            return scales, zeros, max_int

        tmp = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)  # (N_filter,1)
        min_val = torch.minimum(x.amin(dim=1, keepdim=True), tmp)  # (N_filter,1)
        max_val = torch.maximum(x.amax(dim=1, keepdim=True), tmp)  # (N_filter,1)
        
        tmp = (min_val == 0) & (max_val == 0)
        min_val[tmp] = -1
        max_val[tmp] =  1

        scales = (max_val - min_val).clamp(min=1e-5) / max_int  # (N_filter,1)
        zeros = torch.round(-min_val / scales)  # (N_filter,1)

        if self.mse:
            grid = 100
            maxshrink = 0.8
            norm = 2.4

            best = torch.full([x.shape[0]], float("inf"), device=x.device, dtype=x.dtype)
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid 
                minv = p * min_val
                maxv = p * max_val

                tmp_scales = (maxv - minv) / max_int
                tmp_zeros = torch.round(-minv / tmp_scales)

                q = self.quantize_vector(x, tmp_scales, tmp_zeros, max_int)
                q = (q - tmp_zeros) * tmp_scales # dequantize

                q -= x
                q.abs_()
                q.pow_(norm)
                err = torch.sum(q, 1)

                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    scales[tmp] = tmp_scales[tmp]
                    zeros[tmp] = tmp_zeros[tmp]

        return scales, zeros, max_int

    @staticmethod
    def quantize_vector(x, scales, zeros, max_int):
        """
        Quantize a tensor using the given quantization parameteres.
        """
        # quantize
        if max_int == 1:
            x = torch.where(x >= 0, 1, 0)
        else:
            x = torch.clamp(torch.round(x / scales) + zeros, 0, max_int)

        return x

    def quantize(self):
        """
        Perform GPTQ quantization for the weight tensor.
        """
        device = self.W.device

        if self.nbits == 0:
            num_groups = self.W.numel() // self.groupsize if self.groupsize > 0 else self.W.shape[0]
            Q = torch.zeros_like(self.W).reshape(num_groups, -1)
            scales = torch.ones(num_groups, 1, dtype=self.W.dtype, device=device)
            zeros = torch.zeros(num_groups, 1, dtype=self.W.dtype, device=device)
            return Q, scales, zeros
        
        
        W = self.W.float()  # (n_fltd, n_dim)
        H = self.H          # (n_dim, n_dim)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        assert not self.static_groups
        if self.static_groups:
            pass

        assert not self.actorder
        if self.actorder:
            pass

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # initialize quantization params (assuming groupsize < 0)
        scales, zeros, max_int = self.find_params(W)
        scales_list = []
        zeros_list = []

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.groupsize > 0:
                    if not self.static_groups:
                        # update quantization params for a new group
                        if (i1 + i) % self.groupsize == 0:
                            scales, zeros, max_int = self.find_params(W[:, (i1 + i):(i1 + i + self.groupsize)])
                            scales_list.append(scales)
                            zeros_list.append(zeros)
                    else:
                        pass

                q = self.quantize_vector(w.unsqueeze(1), scales, zeros, max_int)
                qr = (q - zeros) * scales  # dequantize

                q = q.flatten()
                qr = qr.flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - qr) ** 2 / d ** 2
                err1 = (w - qr) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.actorder:
            pass

        if self.groupsize != -1:
            scales = torch.cat(scales_list, dim=1).to(self.W.dtype)  # (N_filter, N_group)
            zeros = torch.cat(zeros_list, dim=1).to(self.W.dtype)  # (N_filter, N_group)

            # reformat
            Q = Q.reshape(-1, self.groupsize) # (N_group, groupsize)
            scales = scales.reshape(-1, 1)    # (N_group, 1)
            zeros = zeros.reshape(-1, 1)      # (N_group, 1)

        Q = Q.to(self.W.dtype)
        scales = scales.to(self.W.dtype)
        zeros = zeros.to(self.W.dtype)

        return Q, scales, zeros

    def dequantize(self, Q, scales, zeros):
        """
        Dequantize a quantized tensor using the given quantization parameters.
        """
        return (Q - zeros) * scales
