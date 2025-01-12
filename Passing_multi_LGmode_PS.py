# Модуль состоит из 6 блоков:
# 1. Фурье-преобразование
# 2. Моделирование фазового экрана с заданными параметрами
# 3. Моделирование пучков ЛГ, объединение их в суперпозицию
# 4. Задание параметров построения
# 5. Построение и визуализация


import numpy as np
from numpy import fft
import sympy as smp
import scipy.special
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Функция Фурье-Преобразования. Задаются параметры сетки, осуществляется Фурье с заданными узлами
def ift2(G, delta_k, FFT=None):
    N = G.shape[0]  # количество строк
    M = G.shape[1]  # количество столбцов
    if FFT:
        g = np.fft.fftshift(FFT(np.fft.fftshift(G))) * (N * M)
    else:
        g = fft.ifftshift(fft.ifft2(fft.fftshift(G))) * (N * M)
    return g


#Строим случайный фазовый экран

def AIR_phase_screen(lam, L, C_n_2, N, M, delta, L0, l0, FFT = None, seed=None):
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

    PSD_phi = (0.023 * r0 ** (-5. / 3.) * np.exp(-1 * ((k / km) ** 2)) / (((k ** 2) + (k0 ** 2)) ** (11. / 6)))

    PSD_phi[int(M / 2), int(N / 2)] = 0

    cn = ((R.normal(size=(M, N)) + 1j * R.normal(size=(M, N))) * np.sqrt(PSD_phi * del_k_x * del_k_y))

    phs = ift2(cn, 1, FFT)
    sigmaR = 1.23 * C_n_2 * (2 * np.pi / lam) ** (7 / 6) * L ** (11 / 6)

    return phs



# ---------------------------------------------------Блок 3: Моделирование мод ЛГ, создание их суперпозиции (кубит)-----------------------------------------------------------------
#   В аналитическом виде мод ЛГ произвольного порядека используется факториал, который прописан далее в отдельной функции
def fac(n):
    if n == 1:
        return n
    elif n == 0:
        return 1
    return fac(n - 1) * n


# Функция, отвечающая за создание
def Field(l1, l2, p1, p2, lam, X_size, Y_size, alpha1, x_shift, y_shift, L):  # преобразование LG-пучка в турбулентной среде Field(OAM, x_shift, y_shift, lenghtOfChannel)

    x_0 = np.linspace(-X_size / 2, X_size / 2, X_size) - x_shift  # Диапазон по x в пикселях
    y_0 = np.linspace(-Y_size / 2, Y_size / 2, Y_size) - y_shift  # Диапазон по y в пикселях
    XV, YV = np.meshgrid(x_0, y_0)  # сетка (играет роль декартовых координат теперь)
    #L = L/(1e-8)
    # интенсивность пучка в моде Лагерра-Гаусса записана в цилиндрических координатах, поэтому нужно перейти к декартовым
    k = 2 * np.pi / lam
    # a = 1
    # w0 = ((((k ** 2) * (a ** 2)) / ((k * a ** 2) ** 2 + L ** 2)) ** (0.5)) ** (-1)# здесь получается 250, но это слишком много, можно контролировать, изменяя a
    # Влияет только на размер
    # параметры пучка:

    z = 0  # начало фазового экрана в метрах
    phi_0 = np.arctan2(YV, XV)  # цилиндрическая уловая координата через декартовы
    rho = np.sqrt((np.power(XV, 2) + np.power(YV, 2)))  # цилиндрическая радиальная координата через декартовы
    w0 = 0.24  # перетяжка пучка в z=0
    z0 = np.pi * w0 ** 2 / lam
    w_z = w0 * np.sqrt(2 * (z ** 2 + z0 ** 2) / z0)

    # каждый множитель для выражения по отдельности
    x = 2 * (rho ** 2) / (w_z ** 2)

    # Пучок с l1, p1
    L1_final = scipy.special.assoc_laguerre(x, p1, abs(l1))  # полином Лагерра-Гаусса
    C1 = (2 * fac(p1) / (3.14 * fac(p1 + abs(l1)))) ** (1 / 2)
    q2 = w_z ** (-1)
    q3 = (rho * np.sqrt(2) / w_z) ** (np.abs(l1))
    q4 = np.exp((-1) * (rho / w_z) ** 2)
    q5 = np.exp(((-1j) * k * z * rho ** 2) / (2 * (z0 ** 2 + z ** 2)))
    q6 = np.exp(1j * l1 * phi_0)
    q7 = np.exp((-1j) * (2 * p1 + np.abs(l1) + 1) * np.arctan2(z, z0))

    # Пучок с l2, p2
    L2_final = scipy.special.assoc_laguerre(x, p2, abs(l2))  # полином Лагерра-Гаусса
    C2 = (2 * fac(p2) / (3.14 * fac(p2 + abs(l2)))) ** (1 / 2)
    s2 = w_z ** (-1)
    s3 = (rho * np.sqrt(2) / w_z) ** (np.abs(l2))
    s4 = np.exp((-1) * (rho / w_z) ** 2)
    s5 = np.exp(((-1j) * k * z * rho ** 2) / (2 * (z0 ** 2 + z ** 2)))
    s6 = np.exp(1j * l2 * phi_0)
    s7 = np.exp((-1j) * (2 * p2 + np.abs(l2) + 1) * np.arctan2(z, z0))
    s8 = np.exp(1j * alpha1)

    # собираем всё в суперпозицию и нормируем на максимум
    u1 = C1 * L1_final * q2 * q3 * q4 * q5 * q6 * q7
    u2 = C2 * L2_final * s2 * s3 * s4 * s5 * s6 * s7 * s8

    u = u1 + u2

    u = u / np.max(np.abs(u))
    return u

# -------------------------------------------------------------Блок 4: Параметры построения-----------------------------------------------------------------------------------------------
#  структурный коэффициент турбулентности

nm = 1e-9

lam = 500*nm    #длина волны проходящего через среду света
x_shift = 0     #возможность изменять положение пучка
y_shift = 0     #(0,0) - отцентрирован

#размеры матрицы в пикселях
X_size = 1000
Y_size = 1000

l1 = int(input('Введите азимутальный индекс l1:'))  # OAM первый
p1 = int(input('Введите радиальный индекс p1:'))  # радиальное распределение интенсивности в пучках LG
l2 = int(input('Введите азимутальный индекс l2:'))  # OAM первый
p2 = int(input('Введите радиальный индекс p2:'))  # распределение интенсивности в пучках LG
L = int(input('Введите длину оптического канала L:')) #длина оптического канала в метрах

coifficient_Turb = int(input('Введите силу турбулентности от 13 до 17 Cn:')) #  структурный коэффициент турбулентности

C_n_21 = 10 ** (-coifficient_Turb)

mm = 1e-3
m = 1

L = 500*m             #длина оптического канала в метрах

delta1 = 0.1*mm        #1мм

L0 = 10*m                 # внешний масштаб турбулентности
l0 = 0.04*m            # внутренний масштаб турбулентности
alpha1 = 0 #угол поворота второго пучка относительно первого


# при построении в следующем блоке используются одновременно 3 степени турбулентности
# #генерируем фазовые экраны
Out1 = AIR_phase_screen(lam, L, C_n_21, X_size, Y_size, delta1, L0, l0)   # для слабой


Final_hologram = Field(l1,l2, p1, p2, lam, X_size, Y_size, alpha1, x_shift, y_shift, L) # распределения поля начального состояния пучка
start_Beam = np.abs(Final_hologram)**2  # интенсивность начального состояния

Final_hologram1 = (Final_hologram) * np.exp(1j * Out1)     #связываем поле и фазовый экран
Final_hologram1 = np.abs(Final_hologram1) ** 2    # Интенсивность - квадрат модуля напряжённости
Final_hologram1 = (Final_hologram1) / (np.max(Final_hologram1)) #нормировка на максимум

Out1 = np.angle(Out1)  # распределение фазы после прохождения экрана


# -----------------------------------------------Блок 5: Построение и визуализация-----------------------------------------------------------------------------------------------
# здесь используется построение с помощью функции plt.subplots.
#print('Длина оптического канала =', L, 'm')
#print('Значение азимутального индекса l1 =',l1)
#print('Значение радиального индекса p1 =',l1)

fig, axs = plt.subplots(1, 3, figsize=(10,10))


# начальное состояние поля
ax10 = axs[0].imshow(start_Beam, cmap='gray', aspect='equal',extent=(0, 8, 8, 0), vmin=np.min(start_Beam), vmax=np.max(start_Beam))
axs[0].set_title(f'Input Beam \n LG{p1,l1}+LG{p2,l2}', fontsize=15, family="serif", style="italic", weight="heavy")
axs[0].set_xticks([0,4,8])
axs[0].set_yticks([0,4,8])

#  второй столбец (фазовые экраны)
# выводится вид фазового экрана
ax01 = axs[1].imshow(Out1, cmap='gray', aspect='equal',extent=(0, X_size, Y_size, 0), vmin=np.min(Out1), vmax=np.max(Out1))
axs[1].set_title(f'Phase screen for atmospheric model of turbulance \n \n \n $C^2_n$= {C_n_21}', fontsize=15, family="serif", style="italic", weight="heavy")
axs[1].set_xticks([0,500,1000])
axs[1].set_yticks([0,500,1000])

# третий столбец (результурующий пучок)
# итоговый пучок при слабой турбулентности
ax02 = axs[2].imshow(Final_hologram1, cmap='gray',aspect='equal', extent=(0, X_size, Y_size, 0))
axs[2].set_title(f'Output Beam \n L = {L}m',  fontsize=15, family="serif", style="italic", weight="heavy")
axs[2].set_xticks([0,500,1000])
axs[2].set_yticks([0,500,1000])

plt.show()

plt.subplots_adjust(left = 1,bottom = 1, top = 2, right = 2, hspace = .1, wspace = .2)

#cbar10 = fig.colorbar(ax10, ax=axs[0], ticks=[0, 1], shrink = 0.8)
#cbar10.ax.set_yticklabels(['0','1'])

#cbar01 = fig.colorbar(ax01, ax=axs[1], ticks=[-3.14, 3.14], shrink = 0.3)
#cbar01.ax.set_yticklabels([f'0','2$\pi$'])

#cbar02 = fig.colorbar(ax02, ax=axs[2], ticks=[0, 1], shrink = 0.3)
#cbar02.ax.set_yticklabels(['0','1'])

#fig.savefig('LG-Beams_03_0-3_In_Turb1_L=10m.png',dpi = 500)
