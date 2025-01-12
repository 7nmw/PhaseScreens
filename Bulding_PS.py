'''
   Creates a random phase screen with Von Karmen statistics.
    (Schmidt 2010)
    note::
        The phase screen is returned as a 2d array, with each element representing the phase
        change in **radians**. This means that to obtain the physical phase distortion in nanometres,
        it must be multiplied by (wavelength / (2*pi)), (where wavellength here is the same wavelength
        in which r0 is given in the function arguments)

AIR_phase_screen(lam, L, C_n_2, N, M, delta, L0, l0, FFT=None, seed=None):
lam (nm) -- the wavelength of light passing through the medium in nanometers (длина волны проходящего через среду света в нанометрах)
L (m) (float) -- optical channel length in meters (длина оптического канала в метрах)
C_n_2 (int) -- structural coefficient of turbulence (структурный коэффициент турбулентности)
N, M (int) -- Size of phase scrn in pxls (размеры матрицы в пикселях)
delta (float) -- size in Metres of each pxl (размер пикселя)
L0 (float) --  Size of outer-scale in metres (внешний масштаб турбулентности в метрах)
l0 (float) -- inner scale in metres (внутренний масштаб турбулентности в метрах)

seed (int, optional): seed for random number generator. If provided,
           allows for deterministic screens

r0 (float): r0 parameter of scrn in metres

'''

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt


# Функция Фурье-Преобразования
def ift2(G, FFT=None):
    if FFT:
        g = np.fft.fftshift(FFT(np.fft.fftshift(G)))
    else:
        g = fft.ifftshift(fft.ifft2(fft.fftshift(G)))
    return g


# Строим случайный фазовый экран

def AIR_phase_screen(lam, L, C_n_2, N, M, delta, L0, l0, FFT=None, seed=None):
    # далее, чтобы не возникали ошибки из-за типа переменных, переведём все параметры в float
    delta = float(delta)
    L = float(L)
    L0 = float(L0)
    l0 = float(l0)

    R = np.random.default_rng(
        seed)  # встроенная функция библиотеки, отвечает за случайность величин для построения экрана

    del_k_x = 1. / (delta * N)
    del_k_y = 1. / (delta * M)
    kx = np.arange(-N / 2., N / 2.) * del_k_x
    ky = np.arange(-M / 2., M / 2.) * del_k_y
    (kx, ky) = np.meshgrid(kx, ky)
    k = np.sqrt(kx ** 2 + ky ** 2)
    km = 5.92 / l0
    k0 = 2 * np.pi / L0

    kk = 2 * np.pi / lam
    r0 = 0.185 * (4 * np.pi ** 2 / ((kk) ** 2 * L * C_n_2)) ** (3 / 5)

    PSD_phi = (0.023 * r0 ** (-5. / 3.) * np.exp(-1 * ((k / km) ** 2)) / (
                ((k ** 2) + (k0 ** 2)) ** (11. / 6)))  # modified von Karman atmospheric phase

    PSD_phi[int(M / 2), int(N / 2)] = 0

    cn = ((R.normal(size=(M, N)) + 1j * R.normal(size=(M, N))) * np.sqrt(PSD_phi * del_k_x * del_k_y))

    phs = ift2(cn, FFT).real
    # sigmaR = 1.23 * C_n_2 * (2 * np.pi / lam) ** (7 / 6) * L ** (11 / 6)

    return phs


bb = AIR_phase_screen(500 * 10 ** -9, 100, 10 ** -16, 1000, 1000, 10 ** -4, 10, 0.04)
plt.imshow(bb, cmap='gray')

plt.title('Phase screen for atmospheric model of turbulence \n Weak $C^2_{n}$ = $10^{-16}$', fontsize=15,
          family="serif", style="italic", weight="heavy")
plt.show()
