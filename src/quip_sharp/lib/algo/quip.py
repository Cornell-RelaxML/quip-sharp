import copy
import os

import glog
import torch
from tqdm import tqdm

from quip_sharp.lib import utils


def RHT_H(H, SU):
    return utils.matmul_hadUt(utils.matmul_hadUt(H * SU).T * SU)


def RHT_W(W, SU, SV):
    return utils.matmul_hadUt(utils.matmul_hadUt(W.T * SV).T * SU)


def incoherence_preprocess(H, W, args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    device = H.device
    (m, n) = W.shape

    def _dump(Hr, Lhr, msg=''):
        torch.save(Hr, f"{args.save_pfx}/Hr_debug_fft.pt")
        torch.save(Lhr, f"{args.save_pfx}/Lhr_debug_fft.pt")
        raise Exception(msg)

    # diagonally rescale W,H to minimize proxy loss
    scaleWH = None
    Wr = W
    Hr = H
    if args.rescale_WH:
        Hr = H / H.abs().max()
        diagH = torch.diag(Hr)
        diagW2 = torch.diag(W.T @ W)
        diagH = torch.clamp(diagH, min=1e-8)
        diagW2 = torch.clamp(diagW2, min=1e-8)
        scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
        scaleWH = scaleWH.clamp(min=1e-8)
        Wr = Wr * scaleWH[None, :]
        Hr = Hr / scaleWH[None, :]
        Hr = Hr / scaleWH[:, None]
        scaleWH = scaleWH.cpu()

    # randomized hadamard transformation on H, W
    if args.incoh_mode == "had":
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
        Hr = RHT_H(Hr, SU)
        Wr = RHT_W(Wr, SU, SV)
    # randomized kronecker product on H, W
    elif args.incoh_mode == "kron":
        SU = utils.rand_ortho_butterfly_noblock(n).to(dtype_).to(device)
        SV = utils.rand_ortho_butterfly_noblock(m).to(dtype_).to(device)
        Hr = SU @ Hr @ SU.T
        Wr = SV @ Wr @ SU.T
    else:
        raise NotImplementedError
    SV = SV.cpu()
    SU = SU.cpu()

    Lhr = torch.linalg.cholesky(Hr)
    if not torch.all(torch.isfinite(Lhr)):
        return None

    Wr = Wr.to(device)

    return Lhr, Hr, Wr, SU, SV, scaleWH


def incoherence_process(hatWr, SU, SV, scaleWH, args):
    device = hatWr.device
    # reverse hadamard transformation
    if args.incoh_mode == 'had':
        hatWr = (utils.matmul_hadU(
            (utils.matmul_hadU(hatWr) * SU.to(device)).T) * SV.to(device)).T
    # reverse kronecker product
    elif args.incoh_mode == 'kron':
        hatWr = SV.T.to(device) @ hatWr @ SU.to(device)
    else:
        raise NotImplementedError

    # reverse rescale W,H
    if args.rescale_WH:
        hatWr /= scaleWH[None, :].to(device)

    assert torch.isfinite(hatWr).all()
    return hatWr


def low_rank_preprocess(Wr, Hr, Lhr, args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    if args.full_svd:
        svdZ = torch.linalg.svd(Wr.to(torch.float64) @ Lhr.to(torch.float64),
                                full_matrices=False)
        Hr -= (Lhr.to(torch.float64) @ svdZ.Vh.T[:, :args.lora_rank] @ \
                   svdZ.Vh[:args.lora_rank] @ Lhr.to(torch.float64).T).to(dtype_)
        Hr += torch.diag(Hr).mean() * args.sigma_reg2 * \
            torch.eye(Hr.shape[0], device=Hr.device, dtype=Hr.dtype)
        Wr -= (svdZ.U[:, :args.lora_rank] @ svdZ.U.T[:args.lora_rank] @ Wr.to(
            torch.float64)).to(dtype_)
    else:
        U_lrz, S_lrz, V_lrz = torch.svd_lowrank(
            Wr.to(torch.float64) @ Lhr.to(torch.float64),
            q=2 * args.lora_rank,
            niter=10)
        U_lrz = U_lrz[:, :args.lora_rank]
        V_lrz = V_lrz[:, :args.lora_rank]
        Hr -= (Lhr.to(torch.float64) @ V_lrz @ V_lrz.T @ Lhr.to(
            torch.float64).T).to(dtype_)
        Hr += torch.diag(Hr).mean() * args.sigma_reg2 * \
            torch.eye(Hr.shape[0], device=Hr.device, dtype=Hr.dtype)
        Wr -= (U_lrz @ U_lrz.T @ Wr.to(torch.float64)).to(dtype_)
    return Wr, Hr


def low_rank_process(Wo, hatWr, Lhr, args):
    # invLhr = torch.linalg.inv(Lhr)
    # assert torch.isfinite(invLhr).all()

    svdRZ = torch.linalg.svd((Wo - hatWr) @ Lhr, full_matrices=False)
    A = svdRZ.U[:, :args.lora_rank]
    # B = torch.diag(svdRZ.S[:args.lora_rank]) @ svdRZ.Vh[:args.lora_rank] @ invLhr
    B = torch.linalg.solve_triangular(
        Lhr,
        torch.diag(svdRZ.S[:args.lora_rank]) @ svdRZ.Vh[:args.lora_rank],
        upper=False,
        left=False)
    assert torch.isfinite(A).all() and torch.isfinite(B).all()

    svdB = torch.linalg.svd(B, full_matrices=False)
    A = (A @ svdB.U @ torch.diag(svdB.S.sqrt())).half()
    B = (torch.diag(svdB.S.sqrt()) @ svdB.Vh).half()

    hatWr = hatWr.to(A.device) + \
        (A @ B).to(torch.float64 if args.use_fp64 else torch.float32)
    return hatWr, A, B


def LDLQ(Wr, Hr, L, D, cb, args):
    '''
    want eta = (Wr - hatWr) @ L
    want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
    want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
    Qidxs = torch.zeros(m,
                        n // cb.codesz,
                        dtype=cb.idx_dtype,
                        device=Hr.device)
    for k in reversed(range(n // cb.codesz)):
        WXWX = Wr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + \
            (Wr[:, (cb.codesz * (k + 1)):n] - hatWr[:, (cb.codesz * (k + 1)):n]) @ \
            L[(cb.codesz * (k + 1)):n, (cb.codesz * k):(cb.codesz * (k + 1))]
        hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))], Qidxs[:, k] = \
            cb.quantize(WXWX, resid_scale_override=args.resid_scale_override)
    for ie in range(args.quip_tune_iters):
        for k in reversed(range(n // cb.codesz)):
            WXWX = hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + (Wr - hatWr) @ \
                Hr[:, (cb.codesz * k):(cb.codesz * (k + 1))] @ \
                torch.linalg.inv(Hr[(cb.codesz * k):(cb.codesz * (k + 1)),
                                    (cb.codesz * k):(cb.codesz * (k + 1))])
            hatWr[:, (cb.codesz *
                      k):(cb.codesz * (k + 1))], Qidxs[:, k] = cb.quantize(
                          WXWX, resid_scale_override=args.resid_scale_override)

    return hatWr, Qidxs


def LDLQ_buffered(Wr, Hr, L, D, cb, args, buf_cols=128):
    '''
    reduce overhead of memory r/w
    buffer size is in groups of codesz (4) columns (for D4)
    '''
    (m, n) = Wr.shape
    assert buf_cols % cb.codesz == 0
    assert n % buf_cols == 0
    buf_size = buf_cols // cb.codesz

    hatWr_T = torch.zeros(n, m, dtype=Hr.dtype, device=Hr.device)
    Qidxs_T = torch.zeros(n // cb.codesz,
                          m,
                          dtype=cb.idx_dtype,
                          device=Hr.device)

    device = Wr.device
    Wr = Wr.cpu()
    Hr = Hr.cpu()
    utils.clean()
    Wr_T = Wr.T.contiguous().to(device)
    Hr_T = Hr.T.contiguous().to(device)

    # quip
    prod_cache = torch.zeros(n, m, dtype=Wr_T.dtype, device=Wr_T.device)
    for cur_col in range(n // cb.codesz, 0, -buf_size):
        b_Wr_T = Wr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_L = L[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col].contiguous()
        b_prod = prod_cache[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
        L_offset = cb.codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                b_L[cb.codesz * (i + 1):, L_offset + cb.codesz * i : L_offset + cb.codesz * (i + 1)].T @ \
                (b_Wr_T[cb.codesz * (i + 1):] - b_hatWr_T[cb.codesz * (i + 1):]) + \
                b_prod[cb.codesz * i : cb.codesz * (i + 1)]
            q_out = cb.quantize(WXWX.T,
                                resid_scale_override=args.resid_scale_override)
            b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
            b_Qidxs_T[i] = q_out[1]

        prod_cache += b_L.T @ (b_Wr_T - b_hatWr_T)
        hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col] = b_hatWr_T

    del b_Wr_T, b_hatWr_T, b_L, b_prod, L_offset, prod_cache
    utils.clean()

    # tune
    for ie in range(args.quip_tune_iters):
        # recompute delta to minimize errors
        delta_T = Wr_T - hatWr_T
        for cur_col in range(n // cb.codesz, 0, -buf_size):
            b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Hr_T = Hr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
            b_delta_T = delta_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
            Hr_offset = cb.codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if cb.codesz > 1:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        torch.linalg.inv(b_Hr_T[cb.codesz * i : cb.codesz * (i + 1), Hr_offset + cb.codesz * i : Hr_offset + cb.codesz * (i + 1)].T).T @ b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                else:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        (1/b_Hr_T[i, Hr_offset + i]) * b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] += b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]

                if ie < args.quip_tune_iters - 1:
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = cb.quantize(
                        WXWX.T,
                        return_idx=False,
                        resid_scale_override=args.resid_scale_override).T
                else:
                    q_out = cb.quantize(
                        WXWX.T, resid_scale_override=args.resid_scale_override)
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
                    b_Qidxs_T[i] = q_out[1]

                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] -= b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]
            hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                    cur_col] = b_hatWr_T
            Qidxs_T[cur_col - buf_size:cur_col] = b_Qidxs_T

        del delta_T, b_hatWr_T, b_Hr_T, b_delta_T, b_Qidxs_T, Hr_offset
        utils.clean()

    return hatWr_T.T.contiguous(), Qidxs_T.T.contiguous()


def LDLQ_buffered_lowmem(Wr, Hr, L, D, cb, args, buf_cols=128):
    '''
    reduce overhead of memory r/w
    buffer size is in groups of code_col (4) columns (for D4)
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
    Qidxs = torch.zeros(m,
                        n // cb.codesz,
                        dtype=cb.idx_dtype,
                        device=Hr.device)
    assert n % buf_cols == 0 and buf_cols % cb.codesz == 0
    buf_size = buf_cols // cb.codesz

    # quip
    prod_cache = torch.zeros(m, n, dtype=Wr.dtype, device=Wr.device)
    for cur_col in range(n // cb.codesz, 0, -buf_size):
        b_Wr = Wr[:, cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_hatWr = hatWr[:,
                        cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_L = L[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_prod = prod_cache[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_Qidxs = Qidxs[:, cur_col - buf_size:cur_col]
        L_offset = cb.codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr[:, cb.codesz * i : cb.codesz * (i + 1)] + \
                (b_Wr[:, cb.codesz * (i + 1):] - b_hatWr[:, cb.codesz * (i + 1):]) @ \
                b_L[cb.codesz * (i + 1):, L_offset + cb.codesz * i : L_offset + cb.codesz * (i + 1)] + \
                b_prod[:, cb.codesz * i : cb.codesz * (i + 1)]
            b_hatWr[:, cb.codesz * i:cb.codesz *
                    (i + 1)], b_Qidxs[:, i] = cb.quantize(
                        WXWX, resid_scale_override=args.resid_scale_override)
        prod_cache += (b_Wr - b_hatWr) @ b_L

    del b_Wr, b_hatWr, b_L, b_prod, L_offset, prod_cache
    utils.clean()

    # tune
    for ie in range(args.quip_tune_iters):
        # recompute delta to minimize errors
        delta = Wr - hatWr
        for cur_col in range(n // cb.codesz, 0, -buf_size):
            b_hatWr = hatWr[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
            b_Hr = Hr[:, cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
            b_delta = delta[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
            b_Qidxs = Qidxs[:, cur_col - buf_size:cur_col]
            Hr_offset = cb.codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if cb.codesz > 1:
                    inv = torch.linalg.inv(
                        b_Hr[Hr_offset + cb.codesz * i:Hr_offset + cb.codesz *
                             (i + 1), cb.codesz * i:cb.codesz * (i + 1)])
                else:
                    inv = 1 / b_Hr[Hr_offset + i:Hr_offset + i + 1, i:i + 1]

                WXWX = b_hatWr[:, cb.codesz * i : cb.codesz * (i + 1)] + \
                    delta @ b_Hr[:, cb.codesz * i : cb.codesz * (i + 1)] @ inv

                b_delta[:, cb.codesz * i:cb.codesz *
                        (i + 1)] += b_hatWr[:,
                                            cb.codesz * i:cb.codesz * (i + 1)]

                if ie < args.quip_tune_iters - 1:
                    b_hatWr[:,
                            cb.codesz * i:cb.codesz * (i + 1)] = cb.quantize(
                                WXWX,
                                return_idx=False,
                                resid_scale_override=args.resid_scale_override)
                else:
                    b_hatWr[:, cb.codesz * i:cb.codesz *
                            (i + 1)], b_Qidxs[:, i] = cb.quantize(
                                WXWX,
                                resid_scale_override=args.resid_scale_override)

                b_delta[:, cb.codesz * i:cb.codesz *
                        (i + 1)] -= b_hatWr[:,
                                            cb.codesz * i:cb.codesz * (i + 1)]
        del delta, b_hatWr, b_Hr, b_delta, b_Qidxs, Hr_offset
        utils.clean()

    return hatWr, Qidxs


def quantize(H_orig, W_orig, rank, codebook_orig, args, device='cpu'):
    orig_device = H_orig.device
    W_orig_dtype = W_orig.dtype
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    (m, n) = W_orig.shape

    H = H_orig.clone().to(dtype_).to(device)
    W = W_orig.clone().to(dtype_).to(device)
    codebook = copy.deepcopy(codebook_orig).to(dtype_)

    assert (m % 2 == 0)
    assert (n % 4 == 0)
    assert (torch.all(torch.isfinite(H.cpu())))
    assert (torch.all(torch.isfinite(W.cpu())))

    # incoherence preprocessing
    incoh_out = incoherence_preprocess(H, W, args)
    if incoh_out is None:
        if args.use_fp64:
            raise Exception
        new_args = copy.deepcopy(args)
        new_args.use_fp64 = True
        glog.info('incoherence_preprocess failed, recomputing in fp64')
        del H, W, codebook
        utils.clean()
        return quantize(H_orig, W_orig, rank, codebook_orig, new_args, device)

    Lhr, Hr, Wr, SU, SV, scaleWH = incoh_out
    del incoh_out
    utils.clean()

    glog.info(f'mean square of W: {W.square().mean()}')
    glog.info(f'mean square of Wr: {Wr.square().mean()}')
    glog.info(f'difference between Hr and Hr.T: {((Hr - Hr.T).abs().max())}')
    glog.info(f'max abs of Hr: {((Hr.abs().max()))}')
    glog.info(f'min diag of Lhr: {Lhr.diag().min().item()}')

    Wo = Wr.clone()

    # remove low rank components before LDLQ
    if args.lora_rank > 0:
        Wr, Hr = low_rank_preprocess(Wr, Hr, Lhr, args)

    # block LDL
    block_LDL_out = utils.block_LDL(Hr, codebook.codesz)
    if block_LDL_out is None:
        if args.use_fp64:
            raise Exception
        new_args = copy.deepcopy(args)
        new_args.use_fp64 = True
        glog.info('block_LDL failed, recomputing in fp64')
        del H, W, codebook, Lhr, Hr, Wr, SU, SV, scaleWH, Wo
        utils.clean()
        return quantize(H_orig, W_orig, rank, codebook_orig, new_args, device)

    L, D = block_LDL_out
    del block_LDL_out
    del H_orig, W_orig, codebook_orig
    utils.clean()

    # LDLQ
    Wscale = Wr.square().mean().sqrt()
    if args.scale_override > 0:
        Wscale /= args.scale_override
    else:
        Wscale /= codebook.opt_scale
    Wr = Wr / Wscale
    codebook = codebook.to(device)
    if args.no_use_buffered:
        hatWr, Qidxs = LDLQ(Wr, Hr, L, D, codebook, args)
    elif args.lowmem_ldlq or args.use_fp64:
        hatWr, Qidxs = LDLQ_buffered_lowmem(Wr,
                                            Hr,
                                            L,
                                            D,
                                            codebook,
                                            args,
                                            buf_cols=128)
    else:
        hatWr, Qidxs = LDLQ_buffered(Wr,
                                     Hr,
                                     L,
                                     D,
                                     codebook,
                                     args,
                                     buf_cols=128)

    Wr = Wr.cpu()
    Hr = Hr.cpu()
    L = L.cpu()
    D = D.cpu()
    del Wr, Hr, L, D
    utils.clean()

    hatWr = hatWr * Wscale

    # low rank correction
    if args.lora_rank > 0:
        hatWr, A, B = low_rank_process(Wo, hatWr, Lhr, args)
        A = A.half().cpu()
        B = B.half().cpu()
    else:
        A, B = None, None

    # reverse incoherence process
    hatW = incoherence_process(hatWr, SU, SV, scaleWH, args)

    Qidxs = codebook.maybe_pack_idxs(Qidxs)

    attr = {
        'Qidxs': Qidxs.to(orig_device),
        'A': A,
        'B': B,
        'SU': SU.to(torch.float16).to(orig_device),
        'SV': (SV * Wscale.to(SV.device)).to(
            torch.float16).to(orig_device),  # fuse Wscale into SV
        'scaleWH': scaleWH,
    }

    utils.clean()

    return hatW.to(W_orig_dtype).to(orig_device), attr


def quantize_linear(weights, save_path, hessian_path, cb, args, device='cpu'):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    shapes = [_.shape for _ in weights]
    scales = [_.to(dtype_).square().mean().sqrt() for _ in weights]

    if os.path.exists(save_path):
        return

    H_data = torch.load(hessian_path, map_location=torch.device('cpu'))
    H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
    mu = H_data['mu']
    H.add_(mu[None, :] * mu[:, None])
    n = H_data['n']
    W = torch.vstack([
        weights[i].to(dtype_) / scales[i] for i in range(len(weights))
    ]).to(dtype_)
    H = utils.regularize_H(H, n, args.sigma_reg)
    hatW, attr = quantize(H, W, args.lora_rank, cb, args, device)
    if len(scales) == 1:
        # fuse single scale into SV too
        attr['SV'] *= scales[0]
        scales = [1.0]
    attr.update({
        'fused': len(shapes) > 1,
        'shapes': shapes,
        'scales': scales,
    })
    torch.save(attr, save_path)
    utils.show_metrics(hatW, W, H.to(dtype_), save_path)
    utils.clean()
