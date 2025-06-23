import cmath
import random
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore


# 透镜相位分布
def perfect_lens(pixel_num, pixel_size, focus_length, wavelength):
    x = np.linspace(-pixel_num / 2, pixel_num / 2, pixel_num) * pixel_size
    y = x
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    phase = -2 * np.pi / wavelength * (np.sqrt(r ** 2 + focus_length ** 2) - focus_length)
    return phase

class A_DNN(torch.nn.Module):
    def __init__(self, num_layers, wl, N_pixels, pixel_size, distance, arrangemanet, dz=0., dxy=0., noise_std=0.):
        super(A_DNN, self).__init__()
        self.d = distance
        self.num_layers = num_layers
        self.arrangemanet = arrangemanet

        # 训练中引入的高斯噪声和位移误差的程度
        self.dz = dz
        self.dxy = dxy
        self.noise_std = noise_std
        
        
        self.phase = [torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_pixels, N_pixels)).astype('float64')-0.5)) for _ in range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])

        kz = torch.zeros((N_pixels, N_pixels), dtype=torch.cfloat)
        spectrum_position = torch.linspace(-1 / pixel_size / 2, 1 / pixel_size / 2, N_pixels)
        for i in range(N_pixels):
            for j in range(N_pixels):
                ii = spectrum_position[i]
                jj = spectrum_position[j]
                kz[i, j] = torch.tensor(
                    1j * 2 * np.pi / wl *
                    cmath.sqrt(1 - (wl * ii) ** 2 - (wl * jj) ** 2)
                )
        self.kz=kz
    
    def trasnslation(self, input):
        if len(input.shape) < 4:
            while(len(input.shape) < 4):
                input = input.unsqueeze(0)
        dx = (2*random.random()-1) * self.dxy
        dy = (2*random.random()-1) * self.dxy
        transform_matrix = torch.tensor([
                [1, 0, dx],
                [0, 1 ,dy]]).unsqueeze(0).to(input.device).to(input.dtype)
        
        grid = F.affine_grid(transform_matrix, input.shape)
        output = F.grid_sample(input, grid, mode='nearest')
        return output.squeeze()

    def propagate(self, x, d):
        output = torch.fft.ifft2(
            torch.fft.ifftshift(
                torch.fft.fftshift(
                    torch.fft.fft2(
                        x
                    )
                )
                * torch.exp(self.kz.to(x.device) * d).to(x.device)
            )
        )
        return output
    
    def forward(self, X, stage="train"):
        outpts = []
        assert len(X) == len(self.arrangemanet)
        for x, ar in zip(X, self.arrangemanet):
            x = self.propagate(x, self.d)
            assert len(ar) == self.num_layers
            for index in ar:
                constr_phase = self.phase[index].to(torch.float64)
                phase_noise = torch.randn_like(constr_phase) * self.noise_std

                if stage == "train":
                    constr_phase = constr_phase + phase_noise
                    constr_phase = self.trasnslation(constr_phase)
                    if index == self.num_layers - 1:
                        distance = self.d
                    else:
                        distance = self.d + self.dz * random.random()
                else:
                    constr_phase = constr_phase
                    distance = self.d
                exp_j_phase = torch.exp(1j * constr_phase)
                x = x * exp_j_phase
                x = self.propagate(x, distance)
            x_abs = torch.abs(x) ** 2
            outpts.append(x_abs)
        return outpts