
import torch
from joblib import Parallel, delayed

def torch_trace(input, axis1=0, axis2=1):
    assert input.shape[axis1] == input.shape[axis2], input.shape
    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]
    shape[axis2] = 1
    strides[axis2] = 0
    input = torch.as_strided(input, size=shape, stride=strides)
    return input.sum(dim=(axis1, axis2))


def kernel_delta_norm(device, X_in_1, X_in_2):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    K = torch.zeros((n_1, n_2)).to(device)  # K torch.Size([20, 20])
    u_list = torch.unique(X_in_1)
    for ind in u_list:
        c_1 = torch.sqrt(torch.sum(X_in_1 == ind))
        c_2 = torch.sqrt(torch.sum(X_in_2 == ind))
        ind_1 = torch.where(X_in_1 == ind)[1]  # idx_1 tensor([ 1,  2,  4,  8,  9, 10, 11, 13, 16, 17])
        ind_2 = torch.where(X_in_2 == ind)[1]  # idx_2 tensor([ 1,  2,  4,  8,  9, 10, 11, 13, 16, 17])
        # print('c1',c_1,c_2) # c1 tensor(3.1623) tensor(3.1623)
        K[torch.meshgrid(ind_1, ind_2, indexing='ij')] = 1 / c_1 / c_2
    return K


def rbf_mine(device, X_in_1, X_in_2, sigma=1.0):
    return torch.exp(
        -(torch.cdist(X_in_1.T, X_in_2.T, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2) / (2 * sigma ** 2))


def kernel_gaussian(device, X_in_1, X_in_2, sigma=1.0):
    # return kernels.RBF(length_scale = sigma).__call__(X_in_1.T, X_in_2.T)
    aaa = rbf_mine(device, X_in_1, X_in_2, sigma=1.0)

    return aaa


def make_kernel(device, X, Y, y_kernel, x_kernel='Gaussian', n_jobs=-1, discarded=0, B=0, M=1):
    d, n = X.shape
    #dy = Y.shape[0]
    L = compute_kernel(device, Y, y_kernel, B, M, discarded)
    L = torch.reshape(L, (n * B * M, 1))
    result = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel)(device,
                                                                       torch.reshape(X[k, :], (1, n)), x_kernel, k, B,
                                                                       M, n, discarded) for k in range(d)])
    result = dict(result)
    K = torch.stack([result[k] for k in range(d)]).T
    KtL = torch.matmul(K.T, L)
    return K, KtL, L


def compute_kernel(device, x, kernel, B=0, M=1, discarded=0):
    d, n = x.shape

    H = (torch.eye(B, dtype=torch.float32) - 1 / B * torch.ones(B, dtype=torch.float32)).to(device)
    K = (torch.zeros(n * B * M, dtype=torch.float32)).to(device)

    if kernel in ['Gaussian', 'RationalQuadratic', 'Matern32', 'Matern52', 'ExpSineSquared', 'DotProduct', 'Constant',
                  'Laplacian', 'Periodic']:
        x = (x / (x.std() + 10e-20)).to(torch.float32)

    st = 0
    ed = B ** 2
    for m in range(M):
        torch.manual_seed(m)
        index = torch.randperm(n)
        ####
        for i in range(0, n - discarded, B):
            j = min(n, i + B)
            if kernel == 'Gaussian':
                k = kernel_gaussian(device, x[:, index[i:j]], x[:, index[i:j]], torch.sqrt(torch.Tensor(d)))

            elif kernel == 'Delta':
                k = kernel_delta_norm(device, x[:, index[i:j]], x[:, index[i:j]])
            else:
                raise ValueError('Kernel Error')
            k = torch.tensordot(torch.tensordot(H, k, dims=1), H, dims=1)

            k = k / (torch.linalg.norm(k, 'fro') + 10e-10)
            K[st:ed] = k.flatten()
            st += B ** 2
            ed += B ** 2
    return K


def parallel_compute_kernel(device, x, kernel, feature_idx, B, M, n, discarded):
    return (feature_idx, compute_kernel(device, x, kernel, B, M, discarded))


class Updates():
    def __init__(self, ch_dim, device=1, sigma=1.0, a=1.5, beta=0.0, lam=None, numiter=100, objhowoften=1, tol=1e-5,
                 sigmaknown=False):
        self.ch_dim = ch_dim
        self.device = device
        self.sigma = torch.Tensor([sigma]).to(self.device)
        self.a = torch.Tensor([a]).to(self.device)
        self.beta = torch.Tensor([beta]).to(self.device)
        self.lam = lam  # [inf, 1e-05]
        self.numiter = numiter  # 100
        self.objhowoften = objhowoften  # 1
        self.tol = tol  # 1e-5
        self.sigmaknown = sigmaknown  # False
        self.active_set = None  #

    def compute_obj_fast(self, y, sigmasq, beta, f, zeta, Sigma, trS, wsq, tensor_cal6, tensor_cal7):
        # tensor_cal6 = Kxmu        #  tensor_cal7 = KxTKx 라지시그마       self.a tensor([1.5000], device='cuda:1') 1
        sign_logdet_Sigma, logdet_Sigma = torch.linalg.slogdet(
            Sigma.transpose(2, 0).transpose(2, 1))  # tensor([1]),   tensor([-85.8847])
        obj = 0.5 / sigmasq * torch.sum((y - tensor_cal6) ** 2)  # 1
        + torch.sum((0.5 * f.reshape(-1, 1) * (wsq + trS) + 1) / zeta + (self.a + 0.5) * torch.log(zeta))
        + 0.5 * torch.sum(-sign_logdet_Sigma * logdet_Sigma + tensor_cal7 / sigmasq)  # 345
        # self.a: tensor([1.5])   , self.K: 1,  self.P: 256
        + self.K * (0.5 * self.N * torch.log(2 * torch.pi * sigmasq) - torch.sum((beta + 0.5) * torch.log(f))  # 7 8
                    + self.P * (-(self.a + 1.0) + (self.a + 0.5) * torch.log(self.a + 0.5)  # 6
                                - torch.lgamma(self.a + 0.5) + torch.lgamma(self.a)))
        return obj

    def __proximal_gradient_method(self, w, sigmasq, tensor_cal4_inv, XTy, XTX, absLSS):
        count, epsilon = 1, torch.inf
        w_old = w  # (256,1)
        if (self.active_set is None):
            self.active_set = torch.arange(self.P).to(self.device)  # tensor([0,1,...,256])
        # tensor_cal4_inv = KxTKx + sig ^ 2 * 크사이     식 13 계산 중
        # torch.save(tensor_cal4_inv,'tensor_cal4_inv')
        # exit(0)
        A = torch.diagonal(tensor_cal4_inv[self.active_set][:, self.active_set], dim1=0, dim2=1).T
        # A (256,1)  [[1.00537339], [1.0053742 ], [1.00537464], [1.00537474], ]
        while True:
            B = torch.stack([XTy[self.active_set][:, k] + (torch.eye(len(self.active_set)).to(self.device) - 1) * XTX[
                                                                                                                      self.active_set][
                                                                                                                  :,
                                                                                                                  self.active_set,
                                                                                                                  k] @
                             w_old[self.active_set][:, k]
                             for k in range(self.K)], axis=-1)  # self.K = X.shape[2] # 42000, 256,1  따라서 0
            # B (256,1)
            negative = B < (-sigmasq * self.lam[0] * self.N / absLSS[self.active_set])  # (256,1) [[False], [False],]]
            positive = B > (sigmasq * self.lam[1] * self.N / absLSS[self.active_set])  # (256,1) [[False], [False],]]
            to_zero = ~(negative + positive)

            # #####
            # zero = ~(negative + positive)
            # ch_dim = self.ch_dim
            # cnt = 1
            # ch_sum = 0
            # to_zero = []
            # for i in zero:
            #     if cnt == ch_dim:
            #         if ch_sum == 0:
            #             to_zero.extend([False] * ch_dim)
            #         else:
            #             to_zero.extend([True] * ch_dim)
            #         ch_sum = 0
            #         cnt = 0
            #     ch_sum += i
            #     cnt += 1
            # to_zero = torch.from_numpy(np.reshape(to_zero, (-1, 1)))
            #
            # ####
            w_new_active_set = 1 / A
            w_new_active_set[negative] *= B[negative] + sigmasq * self.lam[0] * self.N / absLSS[self.active_set][
                negative]
            w_new_active_set[positive] *= B[positive] - sigmasq * self.lam[1] * self.N / absLSS[self.active_set][
                positive]
            w_new_active_set[to_zero] = torch.Tensor([0])


            w_new = torch.zeros(w_old.shape).to(self.device)  # (256,1)
            w_new[self.active_set] = w_new_active_set

            count += 1

            epsilon_tmp = torch.linalg.norm(w_new - w_old)
            if ((epsilon - epsilon_tmp) < self.tol):
                if (torch.linalg.norm(w_new) < self.tol):
                    w_new = torch.clone(w_old)
                break
            elif (epsilon_tmp < self.tol):
                if (torch.linalg.norm(w_new) < self.tol):
                    w_new = torch.clone(w_old)
                break
            else:
                epsilon = epsilon_tmp

            w_old = torch.clone(w_new)

        active_set = torch.where(torch.abs(w_new) > self.tol)[0]
            ####
        # cnt = 1
        # active_sum = 0
        # active_set = []
        # for idx, element in enumerate(w_new):
        #     if cnt == ch_dim:
        #         if (torch.mean(active_sum) > self.tol):
        #             active_set.extend(list(range(idx - ch_dim + 1, idx + 1)))
        #         active_sum = 0
        #         cnt = 0
        #     active_sum += torch.abs(element)
        #     cnt += 1
        # # ###

        if (len(active_set) == 0):
            self.active_set = -1
        else:
            self.active_set = torch.unique(torch.where(torch.abs(w_new) > self.tol)[0])
        return w_new

    def fit(self, *, y, X, f_init=None):
        # y=self.Ky  (42000,1), X = self.KX (42000,256,1)
        sigma = self.sigma  # 1
        sigmasq = sigma ** 2  # 1

        self.N, self.P, self.K = X.shape  # 42000, 256,1
        f = torch.ones(self.P).to(self.device) if f_init is None else torch.asarray(f_init).to(
            self.device)  # [1,1,1,..,1] 256
        Sigma = torch.zeros([self.P, self.P, self.K]).to(self.device)  # 256,256,1
        zeta = torch.ones([self.P, self.K]).to(self.device)  # 256,1
        trS = torch.ones([self.P, self.K]).to(self.device)
        wsq = torch.ones([self.P, self.K]).to(self.device)
        w = torch.ones([self.P, self.K]).to(self.device)

        Obj = [float('inf')]
        epsilon = 1
        c = 0

        y, X, beta = torch.asarray(y), torch.asarray(X), torch.asarray(self.beta).to(
            self.device)  # y: (42000,1), X: (42000,256,1), beta: array(0.)
        XTX = torch.matmul(X.transpose(2, 0), X.transpose(2, 0).transpose(2, 1)).transpose(0, 1).transpose(1,
                                                                                                           2)  # (256,256,1)
        XTy = torch.stack([X[:, :, k].T @ y[:, k] for k in range(self.K)], axis=-1)  # (256,1)

        # LSS = torch.stack([torch.linalg.inv(XTX[:, :, k]) @ XTy[:, k] for k in range(self.K)], axis=-1)  # (256,1)
        #####
        results = []
        error_indices = []
        exit=False
        for k in range(self.K):
            try:
                inv_matrix = torch.linalg.inv(XTX[:, :, k])
                result = inv_matrix @ XTy[:, k]
                results.append(result)
            except torch.linalg.LinAlgError:
                exit=True
                error_indices.append(k)

        if exit==True:
            print("Indices with singular matrices:", error_indices)
            exit()
        LSS = torch.stack(results, axis=-1)
        ########

        absLSS = torch.abs(LSS)

        f_save, w_save, sigma_save, bound_save = {}, {}, {}, {}

        while (c <= self.numiter and epsilon > self.tol):
            # 식 23. zeta=s, f=eta, wsq=mu hadamard mu
            zeta = (1 + 0.5 * f.reshape(-1, 1) * (wsq + trS)) / (self.a + 0.5)  # (256,1)
            # 식15 eta 계산
            f = self.K * (1 + 2 * beta) / torch.sum((wsq + trS) / zeta, axis=1)
            # 식 13 계산 중.. tensor_cal4_inv = KxTKx + sig^2 * 크사이
            tensor_cal4_inv = torch.stack([XTX[:, :, k] + sigmasq * torch.diag(f / zeta[:, k]) for k in range(self.K)],
                                          axis=-1)
            tensor_cal4 = torch.stack([torch.linalg.inv(tensor_cal4_inv[:, :, k]) for k in range(self.K)], axis=-1)
            # 식 13 Sigma = 라지 시그마
            Sigma = sigmasq * tensor_cal4

            if (c == 0):
                w = torch.stack(
                    [torch.matmul(torch.matmul(tensor_cal4.transpose(2, 0).transpose(2, 1), X.transpose(2, 0))[k, :, :],
                                  y.T[k, :]) for k
                     in range(self.K)]).T
                c = 1
            else:
                w = self.__proximal_gradient_method(w, sigmasq, tensor_cal4_inv, XTy, XTX, absLSS)  # (256,1)
            # trS = 라지 스그마의 diag
            trS = torch.diagonal(Sigma, dim1=0, dim2=1).T
            # mu hadamard prod mu 를 w**2으로 대체
            wsq = w ** 2

            if (self.sigmaknown == False):  # 식 16
                # tensor_cal6 = Kxmu
                tensor_cal6 = torch.stack(
                    [torch.matmul(X.transpose(2, 0).transpose(2, 1)[k, :, :], w.T[k, :]) for k in range(self.K)]).T
                # KxTKx 라지시그마
                tensor_cal7 = torch_trace(
                    torch.matmul(XTX.transpose(2, 0).transpose(2, 1), Sigma.transpose(2, 0).transpose(2, 1)).transpose(
                        0, 1).transpose(1, 2),
                    axis1=0, axis2=1)
                sigmasq = (torch.sum((y - tensor_cal6) ** 2) + torch.sum(tensor_cal7)) / (self.N * self.K)

            if (c % self.objhowoften == 0):
                if (self.sigmaknown == False):
                    obj = self.compute_obj_fast(y, sigmasq, beta, f, zeta, Sigma, trS, wsq, tensor_cal6, tensor_cal7)
                else:
                    tensor_cal6 = torch.stack(
                        [torch.matmul(X.transpose(2, 0).transpose(2, 1)[k, :, :], w.T[k, :]) for k in range(self.K)]).T
                    tensor_cal7 = torch_trace(
                        torch.matmul(XTX.transpose(0, 2), Sigma.transpose(2, 0).transpose(2, 1)).transpose(0,
                                                                                                           1).transpose(
                            1, 2), axis1=0,
                        axis2=1)
                    obj = self.compute_obj_fast(y, sigmasq, beta, f, zeta, Sigma, trS, wsq, tensor_cal6, tensor_cal7)
            epsilon = torch.abs((Obj[-1] - obj) / obj)
            Obj.append(obj)
            bound = obj + self.K * torch.sum(beta * torch.log(f))
            f_save[str(c)], w_save[str(c)], sigma_save[str(c)], bound_save[str(c)] = f, w, torch.sqrt(sigmasq), bound
            c = c + 1
            if (self.active_set is -1):
                break
        self.fhat_process, self.what_process, self.sigmahat_process, self.bound_process = f_save, w_save, sigma_save, bound_save
        self.fhat, self.what, self.sigmahat, self.bound = f, w, torch.sqrt(sigmasq), bound

        return f, w, torch.sqrt(sigmasq), bound


class Proposed_HSIC_Lasso(object):
    def __init__(self, ch_dim, device, lam, tol=1e-5, nu=1.5, numiter=100, objhowoften=1):
        self.ch_dim = ch_dim
        self.device = device
        self.input_file = None
        self.X_in = None
        self.Y_in = None
        self.KX = None
        self.KXtKy = None
        self.omega = None
        self.A = None
        self.lam = None
        self.featname = None
        self.lam = lam
        self.tol = tol
        self.nu = nu
        self.numiter = numiter
        self.objhowoften = objhowoften

    def input(self, X, Y):
        self.X_in = X.T
        self.Y_in = Y.reshape(1, len(Y))
        return True

    def classification_multi(self, B=20, M=3, n_jobs=-1, kernels=['Gaussian']):
        self._run_hsic_lasso_multi(B=B,
                                   M=M,
                                   n_jobs=n_jobs,
                                   kernels=kernels,
                                   y_kernel='Delta')

    def regression_multi(self, B=20, M=3, n_jobs=-1, kernels=['Gaussian']):
        self._run_hsic_lasso_multi(B=B,
                                   M=M,
                                   n_jobs=n_jobs,
                                   kernels=kernels,
                                   y_kernel='Gaussian')

    def _run_hsic_lasso_multi(self, B, M, n_jobs, kernels=['Gaussian'], y_kernel='Gaussian'):

        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")
        n = self.X_in.shape[1]
        B = torch.tensor(B) if B else n
        numblocks = n / B
        discarded = n % B
        if discarded:
            msg = f'B {B} must be an exact divisor of the number of samples {n}. Number of blocks {numblocks} will be approximated to {int(numblocks)}.'
            numblocks = torch.IntTensor([int(numblocks)]).to(self.device)

        M = 1 + bool(numblocks - 1) * (M - 1)

        K = len(kernels)
        X, Xty, Ky = [], [], []
        for kernel in kernels:
            _X, _Xty, _Ky = make_kernel(self.device, self.X_in, self.Y_in, y_kernel, kernel, n_jobs=n_jobs,
                                        discarded=discarded, B=B, M=M)
            # _X, _Xty, _Ky    (24000,1) , (42000,256),   (256,1)

            X.append(_X)
            Xty.append(_Xty)
            Ky.append(_Ky)
        X = torch.stack(X)
        X = torch.transpose(X, 0, 2)
        X = torch.transpose(X, 0, 1)
        Xty = torch.stack(Xty)
        Xty = torch.transpose(Xty, 0, 2)
        Xty = torch.transpose(Xty, 0, 1)
        Ky = torch.stack(Ky)
        Ky = torch.transpose(Ky, 0, 2)
        Ky = torch.transpose(Ky, 0, 1)[:, 0, :]
        self.KX = X * torch.sqrt(1 / (numblocks * M))  # (42000,245,1)
        self.KXtKy = Xty * 1 / (numblocks * M)  # (256,1,1)
        self.Ky = Ky * torch.sqrt(1 / (numblocks * M))  # (42000,1)

        model = Updates(ch_dim=self.ch_dim, device=self.device, lam=self.lam, tol=self.tol, a=self.nu,
                        numiter=self.numiter, objhowoften=self.objhowoften)
        self.eta, self.what, self.sigma, self.bound = model.fit(y=self.Ky,
                                                                X=self.KX)  # return f, w, np.sqrt(sigmasq), bound
        # self. eta 256,   [2.39968992e+01 3.85607399e+01 4.69873725e+01 1.48194444e+01,
        # self.what (256,1) [[0.16662937], [0.13142591], [0.11904735], [0.21206082], [0.        ],
        # self.sigma 0.0037900163853825525
        # self.bound = -249330.10860718673
        self.eta_process, self.what_process, self.sigma_process, self.bound_process = model.fhat_process, model.what_process, model.sigmahat_process, model.bound_process
        # self.eta_process {dict:11} {'1': array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.
        # self.what_process {dict:11} {'1': array([[ 8.01739477e-02], [ 5.75167870e-02], [ 5.02671422e-02],       dict['1'].shape  (256, 1)
        self.omega = torch.mean(self.what, axis=1)[:, None]
        # self.omega (256,1) [[0.16662937], [0.13142591], [0.11904735], [0.21206082], [0.        ], [0.        ],
        self.A = list(torch.argsort(torch.abs(self.omega).flatten()))[::-1]
        # self.A list:256  [3, 0, 1, 2, 88, 83, 84, 85, 86,
        return True

    # def get_index(self):
    #     return self.A

    def get_index_score(self):
        return torch.argsort(torch.abs(self.omega).flatten(), descending=True), self.omega
    #
    # def get_features(self):
    #     index = self.get_index()
    #     return [self.featname[i] for i in index]