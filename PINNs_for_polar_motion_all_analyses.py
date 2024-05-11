########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "future_figures")):
    os.mkdir(os.path.join(cwd, "future_figures"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2100)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1900, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    "Defines a connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("future_final_pm_PINNs.csv")
lod_data = pd.read_csv("future_final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)

pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y and lod
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

low_lod = 1240  # lower lod index
up_lod = 2380  # higher lod index
low_pm = 0  # lower lod index
up_pm = 2380  # higher lod index
x_data = x[:]
y_data = y[:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp
model_GIA3 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL33 for lod

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2
model_icrv3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 3

model_lod = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod
model_Bary3Ster3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod barystatic and steric excitations

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2
model_cmb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque axial 3

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2
model_icb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque axial 3
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-1,
        max_iter=20)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
ulod = True * 1.
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e0
w_ster = 1e0
w_eq = 1e-3
w_gm = 1e-10
w_gm_geophys = 1e-30

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.05  # relative importance of xp
    coeff2 = 0.01  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.001  # relative importance of xp
    coeff2 = 0.0003  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_10 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_11 = model_icrv3(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_lod(x1)
        M_13 = model_Bary3Ster3(x1)
        M_14 = model_GIA3(x1)
        M_15 = model_cmb1(x1)
        M_16 = model_cmb2(x1)
        M_17 = model_cmb3(x1)
        M_18 = model_icb1(x1)
        M_19 = model_icb2(x1)
        M_20 = model_icb3(x1)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:up_pm, 2:3] + uGrIS * y1[:up_pm, 4:5] + uglac * y1[:up_pm, 6:7] + uTWS * y1[:up_pm,
                                                                                                        8:9] + ueq * y1[
                                                                                                                     :up_pm,
                                                                                                                     14:15]
        tmp_bary2 = uAIS * y1[:up_pm, 3:4] + uGrIS * y1[:up_pm, 5:6] + uglac * y1[:up_pm, 7:8] + uTWS * y1[:up_pm,
                                                                                                        9:10] + ueq * y1[
                                                                                                                      :up_pm,
                                                                                                                      15:16]
        tmp_ster1 = usteric * y1[:up_pm, 12:13]
        tmp_ster2 = usteric * y1[:up_pm, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        tmp_gia3 = Tensor([7.07734755270855e+31 / C_earth]).resize(1, 1)
        tmp_baryster3 = uAIS * y1[low_lod:up_lod, 21:22] + uGrIS * y1[low_lod:up_lod, 22:23] + uglac * y1[
                                                                                                       low_lod:up_lod,
                                                                                                       23:24] + uTWS * y1[
                                                                                                                       low_lod:up_lod,
                                                                                                                       24:25] + \
                        usteric * y1[low_lod:up_lod, 26: 27]
        loss_xp = torch.mean((M_1[:up_pm, 0:1] - y1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[:up_pm, 0:1] - y1[:up_pm, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[:up_pm, 0:1] - tmp_bary1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[:up_pm, 0:1] - tmp_bary2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[:up_pm, 0:1] - tmp_ster1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[:up_pm, 0:1] - tmp_ster2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_lod = torch.mean((M_12[low_lod:up_lod, 0:1] - y1[low_lod:up_lod, 20:21].to(default_device)) ** 2)
        loss_bary3 = torch.mean((M_13[low_lod:up_lod, 0:1] - tmp_baryster3[:, 0:1].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[:up_pm, 0:1] - tmp_gia1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[:up_pm, 0:1] - tmp_gia2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_gia3 = torch.mean((M_14[low_lod:up_lod, 0:1] - tmp_gia3[:, 0:1].to(default_device)) ** 2)
        loss_cmb1 = torch.mean((M_15[:up_pm, 0:1] - y1[:up_pm, 16:17].to(default_device)) ** 2)
        loss_cmb2 = torch.mean((M_16[:up_pm, 0:1] - y1[:up_pm, 17:18].to(default_device)) ** 2)
        loss_cmb3 = torch.mean((M_17[low_lod:up_lod, 0:1] - y1[low_lod:up_lod, 27:28].to(default_device)) ** 2)
        loss_icb1 = torch.mean((M_18[:up_pm, 0:1] - y1[:up_pm, 18:19].to(default_device)) ** 2)
        loss_icb2 = torch.mean((M_19[:up_pm, 0:1] - y1[:up_pm, 19:20].to(default_device)) ** 2)
        loss_icb3 = torch.mean((M_20[low_lod:up_lod, 0:1] - y1[low_lod:up_lod, 28:29].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        lod_geophys = model_lod(x_physics)
        dlod_geophys = torch.autograd.grad(lod_geophys, x_physics, torch.ones_like(lod_geophys), create_graph=True,
                                           allow_unused=True)[0].to(default_device)[:, 0:1]
        bary3_geophys = model_Bary3Ster3(x_physics)
        dbary3_geophys = \
            torch.autograd.grad(bary3_geophys, x_physics, torch.ones_like(bary3_geophys), create_graph=True,
                                allow_unused=True)[0].to(default_device)[:, 0:1]
        geophys_loss_lod = torch.mean(
            (dlod_geophys - (1 + k2prime) / (1 + 4 / 3 * (C_A / C_earth) * (k2 / ks)) * dbary3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)
        gia3_geophys = model_GIA3(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        geophys_loss_gia3 = torch.mean((lod_geophys - e_gia - gia3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism
        cmb1_geophys = model_cmb1(x_physics)
        cmb2_geophys = model_cmb2(x_physics)
        cmb3_geophys = model_cmb3(x_physics)
        icb1_geophys = model_icb1(x_physics)
        icb2_geophys = model_icb2(x_physics)
        icb3_geophys = model_icb3(x_physics)

        dicrv1_geophys = torch.autograd.grad(M_9, x_physics, torch.ones_like(M_9), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_10, x_physics, torch.ones_like(M_10), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv3_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((I_m11 * dxp_geophys + I_m12 * dyp_geophys + I_m13 * dlod_geophys -
                                       (M_9 - xp_geophys) * GammaXtau - cmb1_geophys) ** 2)
        geophys_loss_gm2 = torch.mean((I_m21 * dxp_geophys + I_m22 * dyp_geophys + I_m23 * dlod_geophys -
                                       (M_10 - yp_geophys) * GammaXtau - cmb2_geophys) ** 2)
        geophys_loss_gm3 = torch.mean((I_m31 * dxp_geophys + I_m32 * dyp_geophys + I_m33 * dlod_geophys -
                                       (M_11 - lod_geophys) * GammaXtau - cmb3_geophys) ** 2)

        geophys_loss_gm4 = torch.mean((I_c11 * dicrv1_geophys + I_c12 * dicrv2_geophys + I_c13 * dicrv3_geophys +
                                       (M_9 - xp_geophys) * GammaXtau - icb1_geophys) ** 2)
        geophys_loss_gm5 = torch.mean((I_c21 * dicrv1_geophys + I_c22 * dicrv2_geophys + I_c23 * dicrv3_geophys +
                                       (M_10 - yp_geophys) * GammaXtau - icb2_geophys) ** 2)
        geophys_loss_gm6 = torch.mean((I_c31 * dicrv1_geophys + I_c32 * dicrv2_geophys + I_c33 * dicrv3_geophys +
                                       (M_11 - lod_geophys) * GammaXtau - icb3_geophys) ** 2)

        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_xp * loss_xp + w_yp * loss_yp + w_lod * loss_lod + w_bary * (loss_bary1 + loss_bary2 + loss_bary3) + \
               w_ster * (loss_ster1 + loss_ster2) + \
               w_gia * (loss_gia1 + loss_gia2 + loss_gia3) + w_gm * (
                       loss_cmb1 + loss_cmb2 + loss_cmb3 + loss_icb1 + loss_icb2 + loss_icb3) + \
               w_gm_geophys * (
                       geophys_loss_xp + geophys_loss_yp + geophys_loss_lod + geophys_loss_gia1 + geophys_loss_gia2 +
                       geophys_loss_gia3 + geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                       geophys_loss_gm5 + geophys_loss_gm6)

        loss.backward()
        return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 110  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the eopch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_GIA3.train()
    model_Ster1.train()
    model_Ster2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_icrv3.train()
    model_lod.train()
    model_Bary3Ster3.train()
    model_cmb1.train()
    model_cmb2.train()
    model_cmb3.train()
    model_icb1.train()
    model_icb2.train()
    model_icb3.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[:up_pm, 0:1], y_data[:up_pm, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[:up_pm, 0:1], y_data[:up_pm, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "future_figures/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "future_results_PINNs"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_GIA3.eval()
model_Ster1.eval()
model_Ster2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_icrv3.eval()
model_lod.eval()
model_Bary3Ster3.eval()
model_cmb1.eval()
model_cmb2.eval()
model_cmb3.eval()
model_icb1.eval()
model_icb2.eval()
model_icb3.eval()

xp1 = model_xp(x[up_pm:].to(default_device)).cpu().detach()
yp1 = model_yp(x[up_pm:].to(default_device)).cpu().detach()
########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"prediction_{which_analysis_type}_xp.txt"), xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"prediction_{which_analysis_type}_yp.txt"), yp1)




########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "future_figures_New_GM")):
    os.mkdir(os.path.join(cwd, "future_figures_New_GM"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2100)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1900, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    "Defines a connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("future_final_pm_PINNs.csv")
lod_data = pd.read_csv("future_final_lod_PINNs.csv")

T_icw = np.loadtxt("./results_PINNs_new_GM/T_icw_with_gm.txt")
Q_icw = np.loadtxt("./results_PINNs_new_GM/Q_icw_with_gm.txt")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)

pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y and lod
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

low_lod = 1240  # lower lod index
up_lod = 2380  # higher lod index
low_pm = 0  # lower lod index
up_pm = 2380  # higher lod index
x_data = x[:]
y_data = y[:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2

dtype = torch.FloatTensor
T_icw = Tensor(T_icw).type(dtype).to(default_device)
Q_icw = Tensor(Q_icw).type(dtype).to(default_device)

PIE = Tensor(np.array(np.pi)).float().to(default_device)

w_dynamic = 1.

########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters()), lr=1e-3)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters()), lr=1e-3,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters()), lr=1e-1,
        max_iter=50, tolerance_change=1e-128, tolerance_grad=1e-128)

########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
ulod = True * 1.
use_w_dynamic = True
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e0
w_ster = 1e0
w_eq = 1e-100
w_gm = 1e-10
w_gm_geophys = 1e-30

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.03  # relative importance of xp
    coeff2 = 0.006  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.001  # relative importance of xp
    coeff2 = 0.0003  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_cmb1(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_cmb2(x_physics)  # this only appears in the physical constraints (no data)
        M_15 = model_icb1(x_physics)  # this only appears in the physical constraints (no data)
        M_16 = model_icb2(x_physics)  # this only appears in the physical constraints (no data)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:up_pm, 2:3] + uGrIS * y1[:up_pm, 4:5] + uglac * y1[:up_pm, 6:7] + uTWS * y1[:up_pm,
                                                                                                        8:9] + ueq * y1[
                                                                                                                     :up_pm,
                                                                                                                     14:15]
        tmp_bary2 = uAIS * y1[:up_pm, 3:4] + uGrIS * y1[:up_pm, 5:6] + uglac * y1[:up_pm, 7:8] + uTWS * y1[:up_pm,
                                                                                                        9:10] + ueq * y1[
                                                                                                                      :up_pm,
                                                                                                                      15:16]
        tmp_ster1 = usteric * y1[:up_pm, 12:13]
        tmp_ster2 = usteric * y1[:up_pm, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)

        loss_xp = torch.mean((M_1[:up_pm, 0:1] - y1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[:up_pm, 0:1] - y1[:up_pm, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[:up_pm, 0:1] - tmp_bary1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[:up_pm, 0:1] - tmp_bary2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[:up_pm, 0:1] - tmp_ster1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[:up_pm, 0:1] - tmp_ster2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[:up_pm, 0:1] - y1[:up_pm, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[:up_pm, 0:1] - y1[:up_pm, 15:16].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[:up_pm, 0:1] - tmp_gia1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[:up_pm, 0:1] - tmp_gia2[:up_pm, 0:1].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism
        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((dxp_geophys + 4.2679e-5 * xp_geophys +
                                       1.4511e-2 * yp_geophys + 1.0485e-6 * M_12 - M_13) ** 2)
        geophys_loss_gm2 = torch.mean((dyp_geophys + 4.2679e-5 * yp_geophys -
                                       1.4511e-2 * xp_geophys - 1.0485e-6 * M_11 - M_14) ** 2)

        geophys_loss_gm3 = torch.mean((dicrv1_geophys + torch.divide(PIE, T_icw * Q_icw) * M_11 +
                                       torch.divide(2 * PIE, T_icw) * M_12 + M_15) ** 2)
        geophys_loss_gm4 = torch.mean((dicrv2_geophys + torch.divide(PIE, T_icw * Q_icw) * M_12 -
                                       torch.divide(2 * PIE, T_icw) * M_11 + M_16) ** 2)

        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_dynamic * (w_xp * loss_xp + w_yp * loss_yp +
                            w_bary * (loss_bary1 + loss_bary2) +
                            w_eq * (loss_eq1 + loss_eq2) +
                            w_ster * (loss_ster1 + loss_ster2) +
                            w_gia * (loss_gia1 + loss_gia2) +
                            w_gm_geophys * (
                                    geophys_loss_xp + geophys_loss_yp + geophys_loss_gia1 + geophys_loss_gia2 +
                                    geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4))

        loss.backward()

        print(f"loss: {str(loss.item())}")

    return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 200  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the eopch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_cmb1.train()
    model_cmb2.train()
    model_icb1.train()
    model_icb2.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[:up_pm, 0:1], y_data[:up_pm, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[:up_pm, 0:1], y_data[:up_pm, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "future_figures_New_GM/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "future_results_PINNs_New_GM"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_cmb1.eval()
model_cmb2.eval()
model_icb1.eval()
model_icb2.eval()

xp1 = model_xp(x[up_pm:].to(default_device)).cpu().detach()
yp1 = model_yp(x[up_pm:].to(default_device)).cpu().detach()
########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"prediction_{which_analysis_type}_xp.txt"), xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"prediction_{which_analysis_type}_yp.txt"), yp1)




########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "future_figures")):
    os.mkdir(os.path.join(cwd, "future_figures"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2100)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1900, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    "Defines a connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("future_final_pm_PINNs.csv")
lod_data = pd.read_csv("future_final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)

pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y and lod
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

low_lod = 1240  # lower lod index
up_lod = 2380  # higher lod index
low_pm = 0  # lower lod index
up_pm = 2380  # higher lod index
x_data = x[:]
y_data = y[:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp
model_GIA3 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL33 for lod

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2
model_icrv3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 3

model_lod = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod
model_Bary3Ster3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod barystatic and steric excitations

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2
model_cmb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque axial 3

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2
model_icb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque axial 3
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-1,
        max_iter=20)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
ulod = True * 1.
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e0
w_ster = 1e0
w_eq = 1e-100
w_gm = 1e-10
w_gm_geophys = 1e-30

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.03  # relative importance of xp
    coeff2 = 0.006  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.001  # relative importance of xp
    coeff2 = 0.0003  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_icrv3(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_lod(x1)
        M_15 = model_Bary3Ster3(x1)
        M_16 = model_GIA3(x1)
        M_17 = model_cmb1(x1)
        M_18 = model_cmb2(x1)
        M_19 = model_cmb3(x1)
        M_20 = model_icb1(x1)
        M_21 = model_icb2(x1)
        M_22 = model_icb3(x1)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:up_pm, 2:3] + uGrIS * y1[:up_pm, 4:5] + uglac * y1[:up_pm, 6:7] + uTWS * y1[:up_pm,
                                                                                                        8:9] + ueq * y1[
                                                                                                                     :up_pm,
                                                                                                                     14:15]
        tmp_bary2 = uAIS * y1[:up_pm, 3:4] + uGrIS * y1[:up_pm, 5:6] + uglac * y1[:up_pm, 7:8] + uTWS * y1[:up_pm,
                                                                                                        9:10] + ueq * y1[
                                                                                                                      :up_pm,
                                                                                                                      15:16]
        tmp_ster1 = usteric * y1[:up_pm, 12:13]
        tmp_ster2 = usteric * y1[:up_pm, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        tmp_gia3 = Tensor([7.07734755270855e+31 / C_earth]).resize(1, 1)
        tmp_baryster3 = uAIS * y1[low_lod:up_lod, 21:22] + uGrIS * y1[low_lod:up_lod, 22:23] + uglac * y1[
                                                                                                       low_lod:up_lod,
                                                                                                       23:24] + uTWS * y1[
                                                                                                                       low_lod:up_lod,
                                                                                                                       24:25] + \
                        usteric * y1[low_lod:up_lod, 26: 27]
        loss_xp = torch.mean((M_1[:up_pm, 0:1] - y1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[:up_pm, 0:1] - y1[:up_pm, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[:up_pm, 0:1] - tmp_bary1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[:up_pm, 0:1] - tmp_bary2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[:up_pm, 0:1] - tmp_ster1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[:up_pm, 0:1] - tmp_ster2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[:up_pm, 0:1] - y1[:up_pm, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[:up_pm, 0:1] - y1[:up_pm, 15:16].to(default_device)) ** 2)
        loss_lod = torch.mean((M_14[low_lod:up_lod, 0:1] - y1[low_lod:up_lod, 20:21].to(default_device)) ** 2)
        loss_bary3 = torch.mean((M_15[low_lod:up_lod, 0:1] - tmp_baryster3[:, 0:1].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[:up_pm, 0:1] - tmp_gia1[:up_pm, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[:up_pm, 0:1] - tmp_gia2[:up_pm, 0:1].to(default_device)) ** 2)
        loss_gia3 = torch.mean((M_16[low_lod:up_lod, 0:1] - tmp_gia3[:, 0:1].to(default_device)) ** 2)
        loss_cmb1 = torch.mean((M_17[:up_pm, 0:1] - y1[:up_pm, 16:17].to(default_device)) ** 2)
        loss_cmb2 = torch.mean((M_18[:up_pm, 0:1] - y1[:up_pm, 17:18].to(default_device)) ** 2)
        loss_cmb3 = torch.mean((M_19[low_lod:up_lod, 0:1] - y1[low_lod:up_lod, 27:28].to(default_device)) ** 2)
        loss_icb1 = torch.mean((M_20[:up_pm, 0:1] - y1[:up_pm, 18:19].to(default_device)) ** 2)
        loss_icb2 = torch.mean((M_21[:up_pm, 0:1] - y1[:up_pm, 19:20].to(default_device)) ** 2)
        loss_icb3 = torch.mean((M_22[low_lod:up_lod, 0:1] - y1[low_lod:up_lod, 28:29].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        lod_geophys = model_lod(x_physics)
        dlod_geophys = torch.autograd.grad(lod_geophys, x_physics, torch.ones_like(lod_geophys), create_graph=True,
                                           allow_unused=True)[0].to(default_device)[:, 0:1]
        bary3_geophys = model_Bary3Ster3(x_physics)
        dbary3_geophys = \
            torch.autograd.grad(bary3_geophys, x_physics, torch.ones_like(bary3_geophys), create_graph=True,
                                allow_unused=True)[0].to(default_device)[:, 0:1]
        geophys_loss_lod = torch.mean(
            (dlod_geophys - (1 + k2prime) / (1 + 4 / 3 * (C_A / C_earth) * (k2 / ks)) * dbary3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)
        gia3_geophys = model_GIA3(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        geophys_loss_gia3 = torch.mean((lod_geophys - e_gia - gia3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism
        cmb1_geophys = model_cmb1(x_physics)
        cmb2_geophys = model_cmb2(x_physics)
        cmb3_geophys = model_cmb3(x_physics)
        icb1_geophys = model_icb1(x_physics)
        icb2_geophys = model_icb2(x_physics)
        icb3_geophys = model_icb3(x_physics)

        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv3_geophys = torch.autograd.grad(M_13, x_physics, torch.ones_like(M_13), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((I_m11 * dxp_geophys + I_m12 * dyp_geophys + I_m13 * dlod_geophys -
                                       (M_11 - xp_geophys) * GammaXtau - cmb1_geophys) ** 2)
        geophys_loss_gm2 = torch.mean((I_m21 * dxp_geophys + I_m22 * dyp_geophys + I_m23 * dlod_geophys -
                                       (M_12 - yp_geophys) * GammaXtau - cmb2_geophys) ** 2)
        geophys_loss_gm3 = torch.mean((I_m31 * dxp_geophys + I_m32 * dyp_geophys + I_m33 * dlod_geophys -
                                       (M_13 - lod_geophys) * GammaXtau - cmb3_geophys) ** 2)

        geophys_loss_gm4 = torch.mean((I_c11 * dicrv1_geophys + I_c12 * dicrv2_geophys + I_c13 * dicrv3_geophys +
                                       (M_11 - xp_geophys) * GammaXtau - icb1_geophys) ** 2)
        geophys_loss_gm5 = torch.mean((I_c21 * dicrv1_geophys + I_c22 * dicrv2_geophys + I_c23 * dicrv3_geophys +
                                       (M_12 - yp_geophys) * GammaXtau - icb2_geophys) ** 2)
        geophys_loss_gm6 = torch.mean((I_c31 * dicrv1_geophys + I_c32 * dicrv2_geophys + I_c33 * dicrv3_geophys +
                                       (M_13 - lod_geophys) * GammaXtau - icb3_geophys) ** 2)

        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_xp * loss_xp + w_yp * loss_yp + w_lod * loss_lod +\
               w_bary * (loss_bary1 + loss_bary2 + loss_bary3) + \
               w_ster * (loss_ster1 + loss_ster2) + \
               w_eq * (loss_eq1 + loss_eq2) + \
               w_gia * (loss_gia1 + loss_gia2 + loss_gia3) + w_gm * (
                       loss_cmb1 + loss_cmb2 + loss_cmb3 + loss_icb1 + loss_icb2 + loss_icb3) + \
               w_gm_geophys * (
                       geophys_loss_xp + geophys_loss_yp + geophys_loss_lod + geophys_loss_gia1 + geophys_loss_gia2 +
                       geophys_loss_gia3 + geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                       geophys_loss_gm5 + geophys_loss_gm6)

        loss.backward()
        return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 110  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the eopch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_GIA3.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_icrv3.train()
    model_lod.train()
    model_Bary3Ster3.train()
    model_cmb1.train()
    model_cmb2.train()
    model_cmb3.train()
    model_icb1.train()
    model_icb2.train()
    model_icb3.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[:up_pm, 0:1], y_data[:up_pm, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[:up_pm, 0:1], y_data[:up_pm, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "future_figures/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "future_results_PINNs"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_GIA3.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_icrv3.eval()
model_lod.eval()
model_Bary3Ster3.eval()
model_cmb1.eval()
model_cmb2.eval()
model_cmb3.eval()
model_icb1.eval()
model_icb2.eval()
model_icb3.eval()

xp1 = model_xp(x[up_pm:].to(default_device)).cpu().detach()
yp1 = model_yp(x[up_pm:].to(default_device)).cpu().detach()
########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"prediction_{which_analysis_type}_xp.txt"), xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"prediction_{which_analysis_type}_yp.txt"), yp1)




########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures")):
    os.mkdir(os.path.join(cwd, "figures"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1976, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    "Defines a connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

up_to = - (lod_data.shape[0] - 1520)  # number of training data
x_data = x[up_to:]
y_data = y[up_to:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp
model_GIA3 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL33 for lod

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2
model_icrv3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 3

model_lod = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod
model_Bary3Ster3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod barystatic and steric excitations

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2
model_cmb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque axial 3

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2
model_icb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque axial 3
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-1,
        max_iter=20)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
ulod = True * 1.
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e-3
w_ster = 1e-3
w_eq = 1e-3
w_gm = 1e-10
w_gm_geophys = 1e-3

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.03  # relative importance of xp
    coeff2 = 0.006  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.001  # relative importance of xp
    coeff2 = 0.0007  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_10 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_11 = model_icrv3(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_lod(x1)
        M_13 = model_Bary3Ster3(x1)
        M_14 = model_GIA3(x1)
        M_15 = model_cmb1(x1)
        M_16 = model_cmb2(x1)
        M_17 = model_cmb3(x1)
        M_18 = model_icb1(x1)
        M_19 = model_icb2(x1)
        M_20 = model_icb3(x1)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:, 2:3] + uGrIS * y1[:, 4:5] + uglac * y1[:, 6:7] + uTWS * y1[:, 8:9] + ueq * y1[:, 0:1]
        tmp_bary2 = uAIS * y1[:, 3:4] + uGrIS * y1[:, 5:6] + uglac * y1[:, 7:8] + uTWS * y1[:, 9:10] + ueq * y1[:, 0:1]
        tmp_ster1 = usteric * y1[:, 12:13]
        tmp_ster2 = usteric * y1[:, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        tmp_gia3 = Tensor([7.07734755270855e+31 / C_earth]).resize(1, 1)
        tmp_baryster3 = uAIS * y1[:, 21:22] + uGrIS * y1[:, 22:23] + uglac * y1[:, 23:24] + uTWS * y1[:, 24:25] + \
                        usteric * y1[:, 26: 27]
        loss_xp = torch.mean((M_1[up_to:, 0:1] - y1[up_to:, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[:up_to, 0:1] - y1[:up_to, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[up_to:, 0:1] - y1[up_to:, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[:up_to, 0:1] - y1[:up_to, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[up_to:, 0:1] - tmp_bary1[up_to:, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[up_to:, 0:1] - tmp_bary2[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[up_to:, 0:1] - tmp_ster1[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[up_to:, 0:1] - tmp_ster2[up_to:, 0:1].to(default_device)) ** 2)
        loss_lod = torch.mean((M_12[up_to:, 0:1] - y1[up_to:, 20:21].to(default_device)) ** 2)
        loss_bary3 = torch.mean((M_13[up_to:, 0:1] - tmp_baryster3[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[up_to:, 0:1] - tmp_gia1[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[up_to:, 0:1] - tmp_gia2[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia3 = torch.mean((M_14[up_to:, 0:1] - tmp_gia3[up_to:, 0:1].to(default_device)) ** 2)
        loss_cmb1 = torch.mean((M_15[up_to:, 0:1] - y1[up_to:, 16:17].to(default_device)) ** 2)
        loss_cmb2 = torch.mean((M_16[up_to:, 0:1] - y1[up_to:, 17:18].to(default_device)) ** 2)
        loss_cmb3 = torch.mean((M_17[up_to:, 0:1] - y1[up_to:, 27:28].to(default_device)) ** 2)
        loss_icb1 = torch.mean((M_18[up_to:, 0:1] - y1[up_to:, 18:19].to(default_device)) ** 2)
        loss_icb2 = torch.mean((M_19[up_to:, 0:1] - y1[up_to:, 19:20].to(default_device)) ** 2)
        loss_icb3 = torch.mean((M_20[up_to:, 0:1] - y1[up_to:, 28:29].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        lod_geophys = model_lod(x_physics)
        dlod_geophys = torch.autograd.grad(lod_geophys, x_physics, torch.ones_like(lod_geophys), create_graph=True,
                                           allow_unused=True)[0].to(default_device)[:, 0:1]
        bary3_geophys = model_Bary3Ster3(x_physics)
        dbary3_geophys = \
            torch.autograd.grad(bary3_geophys, x_physics, torch.ones_like(bary3_geophys), create_graph=True,
                                allow_unused=True)[0].to(default_device)[:, 0:1]
        geophys_loss_lod = torch.mean(
            (dlod_geophys - (1 + k2prime) / (1 + 4 / 3 * (C_A / C_earth) * (k2 / ks)) * dbary3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)
        gia3_geophys = model_GIA3(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        geophys_loss_gia3 = torch.mean((lod_geophys - e_gia - gia3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism
        cmb1_geophys = model_cmb1(x_physics)
        cmb2_geophys = model_cmb2(x_physics)
        cmb3_geophys = model_cmb3(x_physics)
        icb1_geophys = model_icb1(x_physics)
        icb2_geophys = model_icb2(x_physics)
        icb3_geophys = model_icb3(x_physics)

        dicrv1_geophys = torch.autograd.grad(M_9, x_physics, torch.ones_like(M_9), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_10, x_physics, torch.ones_like(M_10), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv3_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((I_m11 * dxp_geophys + I_m12 * dyp_geophys + I_m13 * dlod_geophys -
                                       (M_9 - xp_geophys) * GammaXtau - cmb1_geophys) ** 2)
        geophys_loss_gm2 = torch.mean((I_m21 * dxp_geophys + I_m22 * dyp_geophys + I_m23 * dlod_geophys -
                                       (M_10 - yp_geophys) * GammaXtau - cmb2_geophys) ** 2)
        geophys_loss_gm3 = torch.mean((I_m31 * dxp_geophys + I_m32 * dyp_geophys + I_m33 * dlod_geophys -
                                       (M_11 - lod_geophys) * GammaXtau - cmb3_geophys) ** 2)

        geophys_loss_gm4 = torch.mean((I_c11 * dicrv1_geophys + I_c12 * dicrv2_geophys + I_c13 * dicrv3_geophys +
                                       (M_9 - xp_geophys) * GammaXtau - icb1_geophys) ** 2)
        geophys_loss_gm5 = torch.mean((I_c21 * dicrv1_geophys + I_c22 * dicrv2_geophys + I_c23 * dicrv3_geophys +
                                       (M_10 - yp_geophys) * GammaXtau - icb2_geophys) ** 2)
        geophys_loss_gm6 = torch.mean((I_c31 * dicrv1_geophys + I_c32 * dicrv2_geophys + I_c33 * dicrv3_geophys +
                                       (M_11 - lod_geophys) * GammaXtau - icb3_geophys) ** 2)

        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_xp * loss_xp + w_yp * loss_yp + w_lod * loss_lod + w_bary * (loss_bary1 + loss_bary2 + loss_bary3) + \
               w_ster * (loss_ster1 + loss_ster2) + \
               w_gia * (loss_gia1 + loss_gia2 + loss_gia3) + w_gm * (
                       loss_cmb1 + loss_cmb2 + loss_cmb3 + loss_icb1 + loss_icb2 + loss_icb3) + \
               w_gm_geophys * (
                       geophys_loss_xp + geophys_loss_yp + geophys_loss_lod + geophys_loss_gia1 + geophys_loss_gia2 +
                       geophys_loss_gia3 + geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                       geophys_loss_gm5 + geophys_loss_gm6)

        loss.backward()
        return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 150  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the eopch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_GIA3.train()
    model_Ster1.train()
    model_Ster2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_icrv3.train()
    model_lod.train()
    model_Bary3Ster3.train()
    model_cmb1.train()
    model_cmb2.train()
    model_cmb3.train()
    model_icb1.train()
    model_icb2.train()
    model_icb3.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[up_to:, 0:1], y_data[up_to:, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[up_to:, 0:1], y_data[up_to:, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_GIA3.eval()
model_Ster1.eval()
model_Ster2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_icrv3.eval()
model_lod.eval()
model_Bary3Ster3.eval()
model_cmb1.eval()
model_cmb2.eval()
model_cmb3.eval()
model_icb1.eval()
model_icb2.eval()
model_icb3.eval()

xp1 = model_xp(x[:up_to].to(default_device)).cpu().detach()
yp1 = model_yp(x[:up_to].to(default_device)).cpu().detach()
E_xp1 = xp1 - y[:up_to, 0:1]
E_yp1 = yp1 - y[:up_to, 1:2]
########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_xp.txt"), E_xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_yp.txt"), E_yp1)





########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os
from torch.autograd import Variable

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures_new_GM")):
    os.mkdir(os.path.join(cwd, "figures_new_GM"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1976, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    """Defines a fully connected network
    N_INPUT: the dimensionality of input [number of features]
    N_OUTPUT: number of output features
    N_HIDDEN: dimensionality of the hidden space
    N_LAYERS: how many layers
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

up_to = - (lod_data.shape[0] - 1520)  # number of training data
x_data = x[up_to:]
y_data = y[up_to:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2

dtype = torch.FloatTensor
T_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)
Q_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)

PIE = Tensor(np.array(np.pi)).float().to(default_device)

w_dynamic = 1.
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-1,
        max_iter=50, tolerance_change=1e-128, tolerance_grad=1e-128)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
use_w_dynamic = True
save_what_learnt = True
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e-3
w_ster = 1e-3
w_eq = 1e-3
w_gm = 1e-10
w_gm_geophys = 1e-3

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.03 / 5.  # relative importance of xp
    coeff2 = 0.01 / 5.  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.03 / 20.  # relative importance of xp
    coeff2 = 0.01 / 20.  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_cmb1(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_cmb2(x_physics)  # this only appears in the physical constraints (no data)
        M_15 = model_icb1(x_physics)  # this only appears in the physical constraints (no data)
        M_16 = model_icb2(x_physics)  # this only appears in the physical constraints (no data)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:, 2:3] + uGrIS * y1[:, 4:5] + uglac * y1[:, 6:7] + uTWS * y1[:, 8:9] + ueq * M_9[:, 0:1]
        tmp_bary2 = uAIS * y1[:, 3:4] + uGrIS * y1[:, 5:6] + uglac * y1[:, 7:8] + uTWS * y1[:, 9:10] + ueq * M_10[:,
                                                                                                             0:1]
        tmp_ster1 = usteric * y1[:, 12:13]
        tmp_ster2 = usteric * y1[:, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        loss_xp = torch.mean((M_1[up_to:, 0:1] - y1[up_to:, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[:up_to, 0:1] - y1[:up_to, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[up_to:, 0:1] - y1[up_to:, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[:up_to, 0:1] - y1[:up_to, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[up_to:, 0:1] - tmp_bary1[up_to:, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[up_to:, 0:1] - tmp_bary2[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[up_to:, 0:1] - tmp_ster1[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[up_to:, 0:1] - tmp_ster2[up_to:, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[up_to:, 0:1] - y1[up_to:, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[up_to:, 0:1] - y1[up_to:, 15:16].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[up_to:, 0:1] - tmp_gia1[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[up_to:, 0:1] - tmp_gia2[up_to:, 0:1].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism

        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((dxp_geophys + 4.2679e-5 * xp_geophys +
                                       1.4511e-2 * yp_geophys + 1.0485e-6 * M_12 - M_13) ** 2)
        geophys_loss_gm2 = torch.mean((dyp_geophys + 4.2679e-5 * yp_geophys -
                                       1.4511e-2 * xp_geophys - 1.0485e-6 * M_11 - M_14) ** 2)

        geophys_loss_gm3 = torch.mean((dicrv1_geophys + torch.divide(PIE, T_icw * Q_icw) * M_11 +
                                       torch.divide(2 * PIE, T_icw) * M_12 + M_15) ** 2)
        geophys_loss_gm4 = torch.mean((dicrv2_geophys + torch.divide(PIE, T_icw * Q_icw) * M_12 -
                                       torch.divide(2 * PIE, T_icw) * M_11 + M_16) ** 2)

        geophys_loss_gm5 = 2e-5 * (torch.reciprocal(torch.abs(T_icw))) ** 2
        # geophys_loss_gm6 = 1e-2 * (torch.reciprocal(torch.abs(Q_icw))) ** 2
        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_dynamic * (w_xp * loss_xp + w_yp * loss_yp +
                            w_bary * (loss_bary1 + loss_bary2) +
                            w_eq * (loss_eq1 + loss_eq2) +
                            w_ster * (loss_ster1 + loss_ster2) +
                            w_gia * (loss_gia1 + loss_gia2) +
                            w_gm_geophys * (
                                    geophys_loss_xp + geophys_loss_yp + geophys_loss_gia1 + geophys_loss_gia2 +
                                    geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                                    geophys_loss_gm5))

        loss.backward()
        if which_analysis_type == "with_gm":
            print(f"loss: {str(loss.item())} |***|", f"T_icw: {str(T_icw.item() * (x.shape[0] / 365.25))} year |***|",
                  f"Q_icw: {str(Q_icw.item() * (dt_sampling * x.shape[0] / 365.25))}")
        else:
            print(f"loss: {str(loss.item())}")

    return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 200  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the epoch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_cmb1.train()
    model_cmb2.train()
    model_icb1.train()
    model_icb2.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()
        if use_w_dynamic:
            w_dynamic *= 1.5

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[up_to:, 0:1], y_data[up_to:, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[up_to:, 0:1], y_data[up_to:, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures_new_GM/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs_new_GM"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_cmb1.eval()
model_cmb2.eval()
model_icb1.eval()
model_icb2.eval()

xp1 = model_xp(x[:up_to].to(default_device)).cpu().detach()
yp1 = model_yp(x[:up_to].to(default_device)).cpu().detach()
E_xp1 = xp1 - y[:up_to, 0:1]
E_yp1 = yp1 - y[:up_to, 1:2]

########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_xp.txt"), E_xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_yp.txt"), E_yp1)

if which_analysis_type == "with_gm" and save_what_learnt:
    eq1 = model_eq1(x.to(default_device)).cpu().detach()  ## for earthquakes 1
    eq2 = model_eq2(x.to(default_device)).cpu().detach()  ## for earthquakes 2

    icrv1 = model_icrv1(x.to(default_device)).cpu().detach()  ## inner core rotation vector 1
    icrv2 = model_icrv2(x.to(default_device)).cpu().detach()  ## inner core rotation vector 2

    cmb1 = model_cmb1(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 1
    cmb2 = model_cmb2(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 2

    icb1 = model_icb1(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 1
    icb2 = model_icb2(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 2

    T_icw = np.array([T_icw.item() * (x.shape[0] / 365.25)])
    Q_icw = np.array([Q_icw.item() * dt_sampling * x.shape[0] / 365.25])

    np.savetxt(os.path.join(cwd, save_folder_name, f"eq1_{which_analysis_type}.txt"), eq1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"eq2_{which_analysis_type}.txt"), eq2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv1_{which_analysis_type}.txt"), icrv1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv2_{which_analysis_type}.txt"), icrv2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb1_{which_analysis_type}.txt"), cmb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb2_{which_analysis_type}.txt"), cmb2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb1_{which_analysis_type}.txt"), icb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb2_{which_analysis_type}.txt"), icb2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"T_icw_{which_analysis_type}.txt"), T_icw)
    np.savetxt(os.path.join(cwd, save_folder_name, f"Q_icw_{which_analysis_type}.txt"), Q_icw)





########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os
from torch.autograd import Variable

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures_new_GM_different_interval")):
    os.mkdir(os.path.join(cwd, "figures_new_GM_different_interval"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1990, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=2000, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    """Defines a fully connected network
    N_INPUT: the dimensionality of input [number of features]
    N_OUTPUT: number of output features
    N_HIDDEN: dimensionality of the hidden space
    N_LAYERS: how many layers
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)
idx_desired = np.where((time_stamp >= 1990) & (time_stamp <= 2019))[0]
time_stamp = time_stamp[idx_desired]
idx_desired_prediction = np.where((time_stamp >= 1990) & (time_stamp < 2000))[0]
idx_desired_training = np.where((time_stamp >= 2000) & (time_stamp <= 2019))[0]

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[idx_desired, 1:], lod_data.iloc[idx_desired, 1:]], axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]


up_to = idx_desired_training[0]  # number of training data
x_data = x[up_to:]
y_data = y[up_to:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2

dtype = torch.FloatTensor
T_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)
Q_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)

PIE = Tensor(np.array(np.pi)).float().to(default_device)

w_dynamic = 1.
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-1,
        max_iter=50, tolerance_change=1e-128, tolerance_grad=1e-128)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
use_w_dynamic = True
save_what_learnt = True
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e-3
w_ster = 1e-3
w_eq = 1e-3
w_gm = 1e-10
w_gm_geophys = 1e-3

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.01 / 1.  # relative importance of xp
    coeff2 = 0.01 / 1.  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.03 / 20.  # relative importance of xp
    coeff2 = 0.01 / 20.  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_cmb1(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_cmb2(x_physics)  # this only appears in the physical constraints (no data)
        M_15 = model_icb1(x_physics)  # this only appears in the physical constraints (no data)
        M_16 = model_icb2(x_physics)  # this only appears in the physical constraints (no data)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:, 2:3] + uGrIS * y1[:, 4:5] + uglac * y1[:, 6:7] + uTWS * y1[:, 8:9] + ueq * M_9[:, 0:1]
        tmp_bary2 = uAIS * y1[:, 3:4] + uGrIS * y1[:, 5:6] + uglac * y1[:, 7:8] + uTWS * y1[:, 9:10] + ueq * M_10[:,
                                                                                                             0:1]
        tmp_ster1 = usteric * y1[:, 12:13]
        tmp_ster2 = usteric * y1[:, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        loss_xp = torch.mean((M_1[up_to:, 0:1] - y1[up_to:, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[:up_to, 0:1] - y1[:up_to, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[up_to:, 0:1] - y1[up_to:, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[:up_to, 0:1] - y1[:up_to, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[up_to:, 0:1] - tmp_bary1[up_to:, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[up_to:, 0:1] - tmp_bary2[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[up_to:, 0:1] - tmp_ster1[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[up_to:, 0:1] - tmp_ster2[up_to:, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[up_to:, 0:1] - y1[up_to:, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[up_to:, 0:1] - y1[up_to:, 15:16].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[up_to:, 0:1] - tmp_gia1[:, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[up_to:, 0:1] - tmp_gia2[:, 0:1].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism

        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((dxp_geophys + 4.2679e-5 * xp_geophys +
                                       1.4511e-2 * yp_geophys + 1.0485e-6 * M_12 - M_13) ** 2)
        geophys_loss_gm2 = torch.mean((dyp_geophys + 4.2679e-5 * yp_geophys -
                                       1.4511e-2 * xp_geophys - 1.0485e-6 * M_11 - M_14) ** 2)

        geophys_loss_gm3 = torch.mean((dicrv1_geophys + torch.divide(PIE, T_icw * Q_icw) * M_11 +
                                       torch.divide(2 * PIE, T_icw) * M_12 + M_15) ** 2)
        geophys_loss_gm4 = torch.mean((dicrv2_geophys + torch.divide(PIE, T_icw * Q_icw) * M_12 -
                                       torch.divide(2 * PIE, T_icw) * M_11 + M_16) ** 2)

        geophys_loss_gm5 = 2e-5 * (torch.reciprocal(torch.abs(T_icw))) ** 2
        # geophys_loss_gm6 = 1e-2 * (torch.reciprocal(torch.abs(Q_icw))) ** 2
        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_dynamic * (w_xp * loss_xp + w_yp * loss_yp +
                            w_bary * (loss_bary1 + loss_bary2) +
                            w_eq * (loss_eq1 + loss_eq2) +
                            w_ster * (loss_ster1 + loss_ster2) +
                            w_gia * (loss_gia1 + loss_gia2) +
                            w_gm_geophys * (
                                    geophys_loss_xp + geophys_loss_yp + geophys_loss_gia1 + geophys_loss_gia2 +
                                    geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                                    geophys_loss_gm5))

        loss.backward()
        if which_analysis_type == "with_gm":
            print(f"loss: {str(loss.item())} |***|", f"T_icw: {str(T_icw.item() * (x.shape[0] / 365.25))} year |***|",
                  f"Q_icw: {str(Q_icw.item() * (dt_sampling * x.shape[0] / 365.25))}")
        else:
            print(f"loss: {str(loss.item())}")

    return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 200  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the epoch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_cmb1.train()
    model_cmb2.train()
    model_icb1.train()
    model_icb2.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()
        if use_w_dynamic:
            w_dynamic *= 1.5

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[up_to:, 0:1], y_data[up_to:, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[up_to:, 0:1], y_data[up_to:, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures_new_GM_different_interval/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs_new_GM_different_interval"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_cmb1.eval()
model_cmb2.eval()
model_icb1.eval()
model_icb2.eval()

xp1 = model_xp(x[:up_to].to(default_device)).cpu().detach()
yp1 = model_yp(x[:up_to].to(default_device)).cpu().detach()
E_xp1 = xp1 - y[:up_to, 0:1]
E_yp1 = yp1 - y[:up_to, 1:2]

########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_xp.txt"), E_xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_yp.txt"), E_yp1)

if which_analysis_type == "with_gm" and save_what_learnt:
    eq1 = model_eq1(x.to(default_device)).cpu().detach()  ## for earthquakes 1
    eq2 = model_eq2(x.to(default_device)).cpu().detach()  ## for earthquakes 2

    icrv1 = model_icrv1(x.to(default_device)).cpu().detach()  ## inner core rotation vector 1
    icrv2 = model_icrv2(x.to(default_device)).cpu().detach()  ## inner core rotation vector 2

    cmb1 = model_cmb1(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 1
    cmb2 = model_cmb2(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 2

    icb1 = model_icb1(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 1
    icb2 = model_icb2(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 2

    T_icw = np.array([T_icw.item() * (x.shape[0] / 365.25)])
    Q_icw = np.array([Q_icw.item() * dt_sampling * x.shape[0] / 365.25])

    np.savetxt(os.path.join(cwd, save_folder_name, f"eq1_{which_analysis_type}.txt"), eq1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"eq2_{which_analysis_type}.txt"), eq2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv1_{which_analysis_type}.txt"), icrv1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv2_{which_analysis_type}.txt"), icrv2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb1_{which_analysis_type}.txt"), cmb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb2_{which_analysis_type}.txt"), cmb2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb1_{which_analysis_type}.txt"), icb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb2_{which_analysis_type}.txt"), icb2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"T_icw_{which_analysis_type}.txt"), T_icw)
    np.savetxt(os.path.join(cwd, save_folder_name, f"Q_icw_{which_analysis_type}.txt"), Q_icw)




########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os
from torch.autograd import Variable

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures_new_GM_different_interval2")):
    os.mkdir(os.path.join(cwd, "figures_new_GM_different_interval2"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.316
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1990, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1990, d=2010)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    """Defines a fully connected network
    N_INPUT: the dimensionality of input [number of features]
    N_OUTPUT: number of output features
    N_HIDDEN: dimensionality of the hidden space
    N_LAYERS: how many layers
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)
idx_desired = np.where((time_stamp >= 1990) & (time_stamp <= 2019))[0]
time_stamp = time_stamp[idx_desired]
idx_desired_prediction = np.where((time_stamp >= 2010) & (time_stamp <= 2019))[0]
idx_desired_training = np.where((time_stamp >= 1990) & (time_stamp < 2010))[0]

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[idx_desired, 1:], lod_data.iloc[idx_desired, 1:]],
                            axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]

from_to = idx_desired_training[0]  # number of training data
end_to = idx_desired_training[-1]
x_data = x[from_to:end_to]
y_data = y[from_to:end_to]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2

dtype = torch.FloatTensor
T_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)
Q_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)

PIE = Tensor(np.array(np.pi)).float().to(default_device)

w_dynamic = 1.
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-1,
        max_iter=50, tolerance_change=1e-128, tolerance_grad=1e-128)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
use_w_dynamic = True
save_what_learnt = True
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e-3
w_ster = 1e-3
w_eq = 1e-3
w_gm = 1e-10
w_gm_geophys = 1e-3

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.01 / 1.  # relative importance of xp
    coeff2 = 0.01 / 1.  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.03 / 20.  # relative importance of xp
    coeff2 = 0.01 / 20.  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_cmb1(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_cmb2(x_physics)  # this only appears in the physical constraints (no data)
        M_15 = model_icb1(x_physics)  # this only appears in the physical constraints (no data)
        M_16 = model_icb2(x_physics)  # this only appears in the physical constraints (no data)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:, 2:3] + uGrIS * y1[:, 4:5] + uglac * y1[:, 6:7] + uTWS * y1[:, 8:9] + ueq * M_9[:, 0:1]
        tmp_bary2 = uAIS * y1[:, 3:4] + uGrIS * y1[:, 5:6] + uglac * y1[:, 7:8] + uTWS * y1[:, 9:10] + ueq * M_10[:,
                                                                                                             0:1]
        tmp_ster1 = usteric * y1[:, 12:13]
        tmp_ster2 = usteric * y1[:, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        loss_xp = torch.mean((M_1[from_to:end_to, 0:1] - y1[from_to:end_to, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[end_to:, 0:1] - y1[end_to:, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[from_to:end_to, 0:1] - y1[from_to:end_to, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[end_to:, 0:1] - y1[end_to:, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[from_to:end_to, 0:1] - tmp_bary1[from_to:end_to, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[from_to:end_to, 0:1] - tmp_bary2[from_to:end_to, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[from_to:end_to, 0:1] - tmp_ster1[from_to:end_to, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[from_to:end_to, 0:1] - tmp_ster2[from_to:end_to, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[from_to:end_to, 0:1] - y1[from_to:end_to, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[from_to:end_to, 0:1] - y1[from_to:end_to, 15:16].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[from_to:end_to, 0:1] - tmp_gia1[:, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[from_to:end_to, 0:1] - tmp_gia2[:, 0:1].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism

        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((dxp_geophys + 4.2679e-5 * xp_geophys +
                                       1.4511e-2 * yp_geophys + 1.0485e-6 * M_12 - M_13) ** 2)
        geophys_loss_gm2 = torch.mean((dyp_geophys + 4.2679e-5 * yp_geophys -
                                       1.4511e-2 * xp_geophys - 1.0485e-6 * M_11 - M_14) ** 2)

        geophys_loss_gm3 = torch.mean((dicrv1_geophys + torch.divide(PIE, T_icw * Q_icw) * M_11 +
                                       torch.divide(2 * PIE, T_icw) * M_12 + M_15) ** 2)
        geophys_loss_gm4 = torch.mean((dicrv2_geophys + torch.divide(PIE, T_icw * Q_icw) * M_12 -
                                       torch.divide(2 * PIE, T_icw) * M_11 + M_16) ** 2)

        geophys_loss_gm5 = 2e-5 * (torch.reciprocal(torch.abs(T_icw))) ** 2
        # geophys_loss_gm6 = 1e-2 * (torch.reciprocal(torch.abs(Q_icw))) ** 2
        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_dynamic * (w_xp * loss_xp + w_yp * loss_yp +
                            w_bary * (loss_bary1 + loss_bary2) +
                            w_eq * (loss_eq1 + loss_eq2) +
                            w_ster * (loss_ster1 + loss_ster2) +
                            w_gia * (loss_gia1 + loss_gia2) +
                            w_gm_geophys * (
                                    geophys_loss_xp + geophys_loss_yp + geophys_loss_gia1 + geophys_loss_gia2 +
                                    geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                                    geophys_loss_gm5))

        loss.backward()
        if which_analysis_type == "with_gm":
            print(f"loss: {str(loss.item())} |***|", f"T_icw: {str(T_icw.item() * (x.shape[0] / 365.25))} year |***|",
                  f"Q_icw: {str(Q_icw.item() * (dt_sampling * x.shape[0] / 365.25))}")
        else:
            print(f"loss: {str(loss.item())}")

    return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 200  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the epoch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_cmb1.train()
    model_cmb2.train()
    model_icb1.train()
    model_icb2.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()
        if use_w_dynamic:
            w_dynamic *= 1.5

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[from_to:end_to, 0:1], y_data[from_to:end_to, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[from_to:end_to, 0:1], y_data[from_to:end_to, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures_new_GM_different_interval2/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs_new_GM_different_interval2"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_cmb1.eval()
model_cmb2.eval()
model_icb1.eval()
model_icb2.eval()

xp1 = model_xp(x[end_to:].to(default_device)).cpu().detach()
yp1 = model_yp(x[end_to:].to(default_device)).cpu().detach()
E_xp1 = xp1 - y[end_to:, 0:1]
E_yp1 = yp1 - y[end_to:, 1:2]

########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_xp.txt"), E_xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_yp.txt"), E_yp1)

if which_analysis_type == "with_gm" and save_what_learnt:
    eq1 = model_eq1(x.to(default_device)).cpu().detach()  ## for earthquakes 1
    eq2 = model_eq2(x.to(default_device)).cpu().detach()  ## for earthquakes 2

    icrv1 = model_icrv1(x.to(default_device)).cpu().detach()  ## inner core rotation vector 1
    icrv2 = model_icrv2(x.to(default_device)).cpu().detach()  ## inner core rotation vector 2

    cmb1 = model_cmb1(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 1
    cmb2 = model_cmb2(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 2

    icb1 = model_icb1(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 1
    icb2 = model_icb2(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 2

    T_icw = np.array([T_icw.item() * (x.shape[0] / 365.25)])
    Q_icw = np.array([Q_icw.item() * dt_sampling * x.shape[0] / 365.25])

    np.savetxt(os.path.join(cwd, save_folder_name, f"eq1_{which_analysis_type}.txt"), eq1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"eq2_{which_analysis_type}.txt"), eq2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv1_{which_analysis_type}.txt"), icrv1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv2_{which_analysis_type}.txt"), icrv2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb1_{which_analysis_type}.txt"), cmb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb2_{which_analysis_type}.txt"), cmb2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb1_{which_analysis_type}.txt"), icb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb2_{which_analysis_type}.txt"), icb2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"T_icw_{which_analysis_type}.txt"), T_icw)
    np.savetxt(os.path.join(cwd, save_folder_name, f"Q_icw_{which_analysis_type}.txt"), Q_icw)





########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os
from torch.autograd import Variable

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures_new_GM_no_physics")):
    os.mkdir(os.path.join(cwd, "figures_new_GM_no_physics"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1976, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    """Defines a fully connected network
    N_INPUT: the dimensionality of input [number of features]
    N_OUTPUT: number of output features
    N_HIDDEN: dimensionality of the hidden space
    N_LAYERS: how many layers
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

up_to = - (lod_data.shape[0] - 1520)  # number of training data
x_data = x[up_to:]
y_data = y[up_to:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

dtype = torch.FloatTensor
T_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)
Q_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)

PIE = Tensor(np.array(np.pi)).float().to(default_device)

w_dynamic = 1.
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters()), lr=1e-3)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters()), lr=1e-3,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters()), lr=1e-1,
        max_iter=50, tolerance_change=1e-128, tolerance_grad=1e-128)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
use_w_dynamic = True
save_what_learnt = True
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e-3
w_ster = 1e-3
w_eq = 1e-3
w_gm = 1e-10
w_gm_geophys = 1e-3

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

coeff1 = 0.03 / (1.1 * 10000.)  # relative importance of xp
coeff2 = 0.01 / (1.1 * 10000.)  # relative importance of yp
########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)

        loss_xp = torch.mean((M_1[up_to:, 0:1] - y1[up_to:, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[:up_to, 0:1] - y1[:up_to, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[up_to:, 0:1] - y1[up_to:, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[:up_to, 0:1] - y1[:up_to, 1:2].to(default_device)) ** 2)

        # add the losses together
        loss = w_dynamic * (w_xp * loss_xp + w_yp * loss_yp)

        loss.backward()
        print(f"loss: {str(loss.item())}")

    return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 200  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the epoch {i + 1}")
    model_xp.train()
    model_yp.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()
        if use_w_dynamic:
            w_dynamic *= 1.5

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[up_to:, 0:1], y_data[up_to:, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[up_to:, 0:1], y_data[up_to:, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures_new_GM_no_physics/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs_new_GM_no_physics"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()

xp1 = model_xp(x[:up_to].to(default_device)).cpu().detach()
yp1 = model_yp(x[:up_to].to(default_device)).cpu().detach()
E_xp1 = xp1 - y[:up_to, 0:1]
E_yp1 = yp1 - y[:up_to, 1:2]

########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"errors_xp.txt"), E_xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"errors_yp.txt"), E_yp1)





########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os
from torch.autograd import Variable

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures_new_GM")):
    os.mkdir(os.path.join(cwd, "figures_new_GM"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1976, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    """Defines a fully connected network
    N_INPUT: the dimensionality of input [number of features]
    N_OUTPUT: number of output features
    N_HIDDEN: dimensionality of the hidden space
    N_LAYERS: how many layers
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

up_to = - (lod_data.shape[0] - 1520)  # number of training data
x_data = x[up_to:]
y_data = y[up_to:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2

dtype = torch.FloatTensor
T_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)
Q_icw = Variable(torch.randn(1).type(dtype), requires_grad=True).to(default_device)

PIE = Tensor(np.array(np.pi)).float().to(default_device)

w_dynamic = 1.
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-3,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(), T_icw, Q_icw), lr=1e-1,
        max_iter=50, tolerance_change=1e-128, tolerance_grad=1e-128)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
use_w_dynamic = True
save_what_learnt = True
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e0
w_gia = 1e0
w_bary = 1e0
w_ster = 1e0
w_eq = 1e0
w_gm = 1e0
w_gm_geophys = 1e0

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
which_type = "what_learnt"
if which_analysis_type == "with_gm":
    coeff1 = 1.  # relative importance of xp
    coeff2 = 1.  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 1.  # relative importance of xp
    coeff2 = 1.  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_cmb1(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_cmb2(x_physics)  # this only appears in the physical constraints (no data)
        M_15 = model_icb1(x_physics)  # this only appears in the physical constraints (no data)
        M_16 = model_icb2(x_physics)  # this only appears in the physical constraints (no data)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:, 2:3] + uGrIS * y1[:, 4:5] + uglac * y1[:, 6:7] + uTWS * y1[:, 8:9] + ueq * M_9[:, 0:1]
        tmp_bary2 = uAIS * y1[:, 3:4] + uGrIS * y1[:, 5:6] + uglac * y1[:, 7:8] + uTWS * y1[:, 9:10] + ueq * M_10[:,
                                                                                                             0:1]
        tmp_ster1 = usteric * y1[:, 12:13]
        tmp_ster2 = usteric * y1[:, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        loss_xp = torch.mean((M_1[up_to:, 0:1] - y1[up_to:, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[:up_to, 0:1] - y1[:up_to, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[up_to:, 0:1] - y1[up_to:, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[:up_to, 0:1] - y1[:up_to, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[up_to:, 0:1] - tmp_bary1[up_to:, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[up_to:, 0:1] - tmp_bary2[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[up_to:, 0:1] - tmp_ster1[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[up_to:, 0:1] - tmp_ster2[up_to:, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[up_to:, 0:1] - y1[up_to:, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[up_to:, 0:1] - y1[up_to:, 15:16].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[up_to:, 0:1] - tmp_gia1[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[up_to:, 0:1] - tmp_gia2[up_to:, 0:1].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism

        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((dxp_geophys + 4.2679e-5 * xp_geophys +
                                       1.4511e-2 * yp_geophys + 1.0485e-6 * M_12 - M_13) ** 2)
        geophys_loss_gm2 = torch.mean((dyp_geophys + 4.2679e-5 * yp_geophys -
                                       1.4511e-2 * xp_geophys - 1.0485e-6 * M_11 - M_14) ** 2)

        geophys_loss_gm3 = torch.mean((dicrv1_geophys + torch.divide(PIE, T_icw * Q_icw) * M_11 +
                                       torch.divide(2 * PIE, T_icw) * M_12 + M_15) ** 2)
        geophys_loss_gm4 = torch.mean((dicrv2_geophys + torch.divide(PIE, T_icw * Q_icw) * M_12 -
                                       torch.divide(2 * PIE, T_icw) * M_11 + M_16) ** 2)

        geophys_loss_gm5 = 2e-5 * (torch.reciprocal(torch.abs(T_icw))) ** 2
        # geophys_loss_gm6 = 1e-2 * (torch.reciprocal(torch.abs(Q_icw))) ** 2
        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_dynamic * (w_xp * loss_xp + w_yp * loss_yp +
                            w_bary * (loss_bary1 + loss_bary2) +
                            w_eq * (loss_eq1 + loss_eq2) +
                            w_ster * (loss_ster1 + loss_ster2) +
                            w_gia * (loss_gia1 + loss_gia2) +
                            w_gm_geophys * (
                                    geophys_loss_xp + geophys_loss_yp + geophys_loss_gia1 + geophys_loss_gia2 +
                                    geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                                    geophys_loss_gm5))

        loss.backward()
        if which_analysis_type == "with_gm":
            print(f"loss: {str(loss.item())} |***|", f"T_icw: {str(T_icw.item() * (x.shape[0] / 365.25))} year |***|",
                  f"Q_icw: {str(Q_icw.item() * (dt_sampling * x.shape[0] / 365.25))}")
        else:
            print(f"loss: {str(loss.item())}")

    return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 200  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the epoch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_cmb1.train()
    model_cmb2.train()
    model_icb1.train()
    model_icb2.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()
        if use_w_dynamic:
            w_dynamic *= 1.5

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[up_to:, 0:1], y_data[up_to:, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[up_to:, 0:1], y_data[up_to:, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures_new_GM/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs_new_GM"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_cmb1.eval()
model_cmb2.eval()
model_icb1.eval()
model_icb2.eval()

xp1 = model_xp(x[:up_to].to(default_device)).cpu().detach()
yp1 = model_yp(x[:up_to].to(default_device)).cpu().detach()
E_xp1 = xp1 - y[:up_to, 0:1]
E_yp1 = yp1 - y[:up_to, 1:2]

########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_xp.txt"), E_xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_yp.txt"), E_yp1)

if which_analysis_type == "with_gm" and save_what_learnt:
    eq1 = model_eq1(x.to(default_device)).cpu().detach()  ## for earthquakes 1
    eq2 = model_eq2(x.to(default_device)).cpu().detach()  ## for earthquakes 2

    icrv1 = model_icrv1(x.to(default_device)).cpu().detach()  ## inner core rotation vector 1
    icrv2 = model_icrv2(x.to(default_device)).cpu().detach()  ## inner core rotation vector 2

    cmb1 = model_cmb1(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 1
    cmb2 = model_cmb2(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 2

    icb1 = model_icb1(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 1
    icb2 = model_icb2(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 2

    T_icw = np.array([T_icw.item() * (x.shape[0] / 365.25)])
    Q_icw = np.array([Q_icw.item() * dt_sampling * x.shape[0] / 365.25])

    np.savetxt(os.path.join(cwd, save_folder_name, f"eq1_{which_analysis_type}_{which_type}.txt"), eq1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"eq2_{which_analysis_type}_{which_type}.txt"), eq2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv1_{which_analysis_type}_{which_type}.txt"), icrv1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icrv2_{which_analysis_type}_{which_type}.txt"), icrv2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb1_{which_analysis_type}_{which_type}.txt"), cmb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"cmb2_{which_analysis_type}_{which_type}.txt"), cmb2)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb1_{which_analysis_type}_{which_type}.txt"), icb1)
    np.savetxt(os.path.join(cwd, save_folder_name, f"icb2_{which_analysis_type}_{which_type}.txt"), icb2)



########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures")):
    os.mkdir(os.path.join(cwd, "figures_what_learnt"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1976, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    "Defines a connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

up_to = - (lod_data.shape[0] - 1520)  # number of training data
x_data = x[up_to:]
y_data = y[up_to:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp
model_GIA3 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL33 for lod

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2
model_icrv3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 3

model_lod = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod
model_Bary3Ster3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod barystatic and steric excitations

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2
model_cmb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque axial 3

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2
model_icb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque axial 3
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-1,
        max_iter=20)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
ulod = True * 1.
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e0
w_gia = 1e0
w_bary = 1e0
w_ster = 1e0
w_eq = 1e0
w_gm1 = 1e0
w_gm2 = 1e-10
w_gm_geophys = 1e0

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "with_gm"
if which_analysis_type == "with_gm":
    coeff1 = 1.00  # relative importance of xp
    coeff2 = 1.00  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.003  # relative importance of xp
    coeff2 = 0.0009  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_icrv3(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_lod(x1)
        M_15 = model_Bary3Ster3(x1)
        M_16 = model_GIA3(x1)
        M_17 = model_cmb1(x1)
        M_18 = model_cmb2(x1)
        M_19 = model_cmb3(x1)
        M_20 = model_icb1(x1)
        M_21 = model_icb2(x1)
        M_22 = model_icb3(x1)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:, 2:3] + uGrIS * y1[:, 4:5] + uglac * y1[:, 6:7] + uTWS * y1[:, 8:9] + ueq * M_9[:, 0:1]
        tmp_bary2 = uAIS * y1[:, 3:4] + uGrIS * y1[:, 5:6] + uglac * y1[:, 7:8] + uTWS * y1[:, 9:10] + ueq * M_10[:,
                                                                                                             0:1]
        tmp_ster1 = usteric * y1[:, 12:13]
        tmp_ster2 = usteric * y1[:, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        tmp_gia3 = Tensor([7.07734755270855e+31 / C_earth]).resize(1, 1)
        tmp_baryster3 = uAIS * y1[:, 21:22] + uGrIS * y1[:, 22:23] + uglac * y1[:, 23:24] + uTWS * y1[:, 24:25] + \
                        usteric * y1[:, 26: 27]
        loss_xp = torch.mean((M_1[up_to:, 0:1] - y1[up_to:, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[:up_to, 0:1] - y1[:up_to, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[up_to:, 0:1] - y1[up_to:, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[:up_to, 0:1] - y1[:up_to, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[up_to:, 0:1] - tmp_bary1[up_to:, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[up_to:, 0:1] - tmp_bary2[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[up_to:, 0:1] - tmp_ster1[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[up_to:, 0:1] - tmp_ster2[up_to:, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[up_to:, 0:1] - y1[up_to:, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[up_to:, 0:1] - y1[up_to:, 15:16].to(default_device)) ** 2)
        loss_lod = torch.mean((M_14[up_to:, 0:1] - y1[up_to:, 20:21].to(default_device)) ** 2)
        loss_bary3 = torch.mean((M_15[up_to:, 0:1] - tmp_baryster3[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[up_to:, 0:1] - tmp_gia1[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[up_to:, 0:1] - tmp_gia2[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia3 = torch.mean((M_16[up_to:, 0:1] - tmp_gia3[up_to:, 0:1].to(default_device)) ** 2)
        loss_cmb1 = torch.mean((M_17[up_to:, 0:1] - y1[up_to:, 16:17].to(default_device)) ** 2)
        loss_cmb2 = torch.mean((M_18[up_to:, 0:1] - y1[up_to:, 17:18].to(default_device)) ** 2)
        loss_cmb3 = torch.mean((M_19[up_to:, 0:1] - y1[up_to:, 27:28].to(default_device)) ** 2)
        loss_icb1 = torch.mean((M_20[up_to:, 0:1] - y1[up_to:, 18:19].to(default_device)) ** 2)
        loss_icb2 = torch.mean((M_21[up_to:, 0:1] - y1[up_to:, 19:20].to(default_device)) ** 2)
        loss_icb3 = torch.mean((M_22[up_to:, 0:1] - y1[up_to:, 28:29].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        lod_geophys = model_lod(x_physics)
        dlod_geophys = torch.autograd.grad(lod_geophys, x_physics, torch.ones_like(lod_geophys), create_graph=True,
                                           allow_unused=True)[0].to(default_device)[:, 0:1]
        bary3_geophys = model_Bary3Ster3(x_physics)
        dbary3_geophys = \
            torch.autograd.grad(bary3_geophys, x_physics, torch.ones_like(bary3_geophys), create_graph=True,
                                allow_unused=True)[0].to(default_device)[:, 0:1]
        geophys_loss_lod = torch.mean(
            (dlod_geophys - (1 + k2prime) / (1 + 4 / 3 * (C_A / C_earth) * (k2 / ks)) * dbary3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)
        gia3_geophys = model_GIA3(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        geophys_loss_gia3 = torch.mean((lod_geophys - e_gia - gia3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism
        cmb1_geophys = model_cmb1(x_physics)
        cmb2_geophys = model_cmb2(x_physics)
        cmb3_geophys = model_cmb3(x_physics)
        icb1_geophys = model_icb1(x_physics)
        icb2_geophys = model_icb2(x_physics)
        icb3_geophys = model_icb3(x_physics)

        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv3_geophys = torch.autograd.grad(M_13, x_physics, torch.ones_like(M_13), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((I_m11 * dxp_geophys + I_m12 * dyp_geophys + I_m13 * dlod_geophys -
                                       (M_11 - xp_geophys) * GammaXtau - cmb1_geophys) ** 2)
        geophys_loss_gm2 = torch.mean((I_m21 * dxp_geophys + I_m22 * dyp_geophys + I_m23 * dlod_geophys -
                                       (M_12 - yp_geophys) * GammaXtau - cmb2_geophys) ** 2)
        geophys_loss_gm3 = torch.mean((I_m31 * dxp_geophys + I_m32 * dyp_geophys + I_m33 * dlod_geophys -
                                       (M_13 - lod_geophys) * GammaXtau - cmb3_geophys) ** 2)

        geophys_loss_gm4 = torch.mean((I_c11 * dicrv1_geophys + I_c12 * dicrv2_geophys + I_c13 * dicrv3_geophys +
                                       (M_11 - xp_geophys) * GammaXtau - icb1_geophys) ** 2)
        geophys_loss_gm5 = torch.mean((I_c21 * dicrv1_geophys + I_c22 * dicrv2_geophys + I_c23 * dicrv3_geophys +
                                       (M_12 - yp_geophys) * GammaXtau - icb2_geophys) ** 2)
        geophys_loss_gm6 = torch.mean((I_c31 * dicrv1_geophys + I_c32 * dicrv2_geophys + I_c33 * dicrv3_geophys +
                                       (M_13 - lod_geophys) * GammaXtau - icb3_geophys) ** 2)

        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_xp * loss_xp + w_yp * loss_yp + w_lod * loss_lod + \
               w_bary * (loss_bary1 + loss_bary2 + loss_bary3) + \
               w_eq * (loss_eq1 + loss_eq2) + \
               w_ster * (loss_ster1 + loss_ster2) + \
               w_gia * (loss_gia1 + loss_gia2 + loss_gia3) + w_gm1 * (
                       loss_cmb1 + loss_cmb2 + loss_cmb3) + w_gm2 * (loss_icb1 + loss_icb2 + loss_icb3) + \
               w_gm_geophys * (
                       geophys_loss_xp + geophys_loss_yp + geophys_loss_lod + geophys_loss_gia1 + geophys_loss_gia2 +
                       geophys_loss_gia3 + geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                       geophys_loss_gm5 + geophys_loss_gm6)

        loss.backward()
        return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 1000  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the eopch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_GIA3.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_icrv3.train()
    model_lod.train()
    model_Bary3Ster3.train()
    model_cmb1.train()
    model_cmb2.train()
    model_cmb3.train()
    model_icb1.train()
    model_icb2.train()
    model_icb3.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[up_to:, 0:1], y_data[up_to:, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[up_to:, 0:1], y_data[up_to:, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures_what_learnt/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs_what_learnt"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_GIA3.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_icrv3.eval()
model_lod.eval()
model_Bary3Ster3.eval()
model_cmb1.eval()
model_cmb2.eval()
model_cmb3.eval()
model_icb1.eval()
model_icb2.eval()
model_icb3.eval()
########################################################################################################################
## what has the network learnt?

eq1 = model_eq1(x.to(default_device)).cpu().detach()  ## for earthquakes 1
eq2 = model_eq2(x.to(default_device)).cpu().detach()  ## for earthquakes 2

icrv1 = model_icrv1(x.to(default_device)).cpu().detach()  ## inner core rotation vector 1
icrv2 = model_icrv2(x.to(default_device)).cpu().detach()  ## inner core rotation vector 2
icrv3 = model_icrv3(x.to(default_device)).cpu().detach()  ## inner core rotation vector 3

cmb1 = model_cmb1(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 1
cmb2 = model_cmb2(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque equatorial 2
cmb3 = model_cmb3(x.to(default_device)).cpu().detach() / nfft  ## core mantle boundary torque axial 3

icb1 = model_icb1(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 1
icb2 = model_icb2(x.to(default_device)).cpu().detach() / nfft  ## inner core torque equatorial 2
icb3 = model_icb3(x.to(default_device)).cpu().detach() / nfft  ## inner core torque axial 3
########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"eq1_{which_analysis_type}.txt"), eq1)
np.savetxt(os.path.join(cwd, save_folder_name, f"eq2_{which_analysis_type}.txt"), eq2)

np.savetxt(os.path.join(cwd, save_folder_name, f"icrv1_{which_analysis_type}.txt"), icrv1)
np.savetxt(os.path.join(cwd, save_folder_name, f"icrv2_{which_analysis_type}.txt"), icrv2)
np.savetxt(os.path.join(cwd, save_folder_name, f"icrv3_{which_analysis_type}.txt"), icrv3)

np.savetxt(os.path.join(cwd, save_folder_name, f"cmb1_{which_analysis_type}.txt"), cmb1)
np.savetxt(os.path.join(cwd, save_folder_name, f"cmb2_{which_analysis_type}.txt"), cmb2)
np.savetxt(os.path.join(cwd, save_folder_name, f"cmb3_{which_analysis_type}.txt"), cmb3)

np.savetxt(os.path.join(cwd, save_folder_name, f"icb1_{which_analysis_type}.txt"), icb1)
np.savetxt(os.path.join(cwd, save_folder_name, f"icb2_{which_analysis_type}.txt"), icb2)
np.savetxt(os.path.join(cwd, save_folder_name, f"icb3_{which_analysis_type}.txt"), icb3)





########################################################################################################################
## step 1: loading the required packages
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
import os

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "figures")):
    os.mkdir(os.path.join(cwd, "figures"))

########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)

########################################################################################################################
########################################################################################################################
## define some constants

C_A = 2.6068e35  # C-A of earth (difference between mean equatorial and axial components of the earth inertia tensor)
# kgm^2
C_earth = 8.0345e37  # third component of inertia tensor of the earth kgm^2
Omega = 7.2921e-5  # rad per second: rad/second
rho_w = 1000  # density of water kg/m^3
h2 = 0.6149
k2 = 0.3055  # tidal love number
k2prime = -0.3
ks = 0.942  # fluid love number

## inertia tensor of the mantle: normalize for numerical purposes: unitless
I_m11 = 7.073257e37 / C_earth
I_m12 = 6.328063e33 / C_earth
I_m13 = 3.969441e32 / C_earth
I_m21 = 6.328063e33 / C_earth
I_m22 = 7.073205e37 / C_earth
I_m23 = -3.093338e32 / C_earth
I_m31 = 3.969441e32 / C_earth
I_m32 = -3.093338e32 / C_earth
I_m33 = 7.097067e37 / C_earth

## inertia tensor of the inner core: normalize for numerical purposes: unitless
I_c11 = 5.852133e34 / C_earth
I_c12 = -1.382824e28 / C_earth
I_c13 = -2.316297e30 / C_earth
I_c21 = -1.382824e28 / C_earth
I_c22 = 5.852130e34 / C_earth
I_c23 = 8.430630e29 / C_earth
I_c31 = -2.316297e30 / C_earth
I_c32 = 8.430630e29 / C_earth
I_c33 = 5.866250e34 / C_earth

## fundamental torque of the geomagnetism
Gamma = 1e21  # Nm unit: Newton-meter
tau = 10 * 365.25 * 86400  # viscous relaxation time in seconds

GammaXtau = Gamma / C_earth * tau / Omega  # multiplication of Gamma and tau: unitless (perhaps per radian)

nfft = 1 / C_earth * (1 / Omega ** 2)  # normalizing factor for the electromagnetic torque

dt_sampling = 18.25  # smapling rate of the pm and lod

ni_start = 0.  ## starting point of the normalized interval time_stamp --> [ni_start, ni_end]
ni_end = 10.  ## ending point of the normalized interval time_stamp --> [ni_start, ni_end]


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define saving plot data functions

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps), loop=loop)


def convert_interval(x, a, b, c, d):
    x = ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b))
    return x


def plot_result(x, y, x_data, y_data, yh, xp=None, plot_title=False, plot_ylabel=[], plot_xlabel=False):
    "Pretty plot training results"
    # plt.figure(figsize=(12,8))
    x_converted = convert_interval(x=x, a=ni_start, b=ni_end, c=1900, d=2019)
    x_data_converted = convert_interval(x=x_data, a=x_data[0], b=x_data[-1], c=1976, d=2019)
    plt.plot(x_converted, y, color="grey", linewidth=2, alpha=0.8, label="Interannual polar motion")
    plt.plot(x_converted, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data_converted, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")

    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-1.1, 1.1)
    if plot_title:
        plt.title("Optimizer: " + OPT + "; Training epoch: %i" % (i + 1), fontsize="large", color="k")
    if len(plot_ylabel) != 0:
        plt.ylabel(plot_ylabel + " [as]")
    if plot_xlabel:
        plt.xlabel("time [year]")

    plt.grid()
    # plt.axis("off")


########################################################################################################################
########################################################################################################################
# define MLP neural networks, which are the basis of our work
class FCN(nn.Module):
    """Defines a fully connected network
    N_INPUT: the dimensionality of input [number of features]
    N_OUTPUT: number of output features
    N_HIDDEN: dimensionality of the hidden space
    N_LAYERS: how many layers
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## read the data
pm_data = pd.read_csv("final_pm_PINNs.csv")
lod_data = pd.read_csv("final_lod_PINNs.csv")
## normalize the torque values so that the machine learning works
pm_data[["geomagnetic_1_CMB", "geomagnetic_2_CMB", "geomagnetic_1_ICB", "geomagnetic_2_ICB"]] *= nfft
lod_data[["geomagnetic_3_CMB", "geomagnetic_3_ICB"]] *= nfft

## normalize the time
time_stamp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

T_chandler = (ni_end - ni_start) * (433 / dt_sampling) / time_stamp.shape[0]
T_annual = (ni_end - ni_start) * (365.25 / dt_sampling) / time_stamp.shape[0]
T_Markowitz = (ni_end - ni_start) * (365.25 * 30 / dt_sampling) / time_stamp.shape[0]
T_core = (ni_end - ni_start) * (365.25 * 6.6 / dt_sampling) / time_stamp.shape[0]

time_stamp = convert_interval(x=time_stamp, a=time_stamp[0], b=time_stamp[-1], c=ni_start, d=ni_end)

time_stamp = np.concatenate((time_stamp,
                             np.cos(2 * np.pi * time_stamp / T_Markowitz),
                             np.sin(2 * np.pi * time_stamp / T_Markowitz)),
                            axis=1)
# time_stamp = np.concatenate((time_stamp, np.cos(2 * np.pi * time_stamp / T_chandler),
#                              np.sin(2 * np.pi * time_stamp / T_chandler),
#                              np.cos(2 * np.pi * time_stamp / T_annual),
#                              np.sin(2 * np.pi * time_stamp / T_annual),
#                              np.cos(2 * np.pi * time_stamp / T_Markowitz),
#                              np.sin(2 * np.pi * time_stamp / T_Markowitz)),
#                             axis=1)
pm_lod_together = pd.concat([pm_data.iloc[:, 1:], lod_data.iloc[:, 1:]], axis=1)  ## concatenate x and y
########################################################################################################################
########################################################################################################################
x = Tensor(time_stamp).float()
y = Tensor(pm_lod_together.values).float() / 1e3
print("We have " + str(time_stamp.shape[0]) + " values in total!")
########################################################################################################################
########################################################################################################################
n = time_stamp.shape[0]
tmp = pm_data["date"].values.reshape(pm_data.shape[0], 1)

up_to = - (lod_data.shape[0] - 1520)  # number of training data
x_data = x[up_to:]
y_data = y[up_to:]
print(x_data.shape, y_data.shape)
########################################################################################################################
########################################################################################################################
## plot or not?

plot_or_not = False

if plot_or_not:
    plt.figure()
    plt.plot(x[:, 0:1], y[:, 0:1], color="tab:blue", label="Exact solution x_p [as]")
    plt.plot(x[:, 0:1], y[:, 1:2], color="tab:orange", label="Exact solution y_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 0:1], color="tab:pink", label="Training data x_p [as]")
    plt.scatter(x_data[:, 0:1], y_data[:, 1:2], color="tab:green", label="Training data y_p [as]")
    plt.plot(x[:, 0:1], y[:, 2:3], color="tab:cyan", label="Antarct Ice Sheet $\psi_1$ [as]")
    plt.plot(x[:, 0:1], y[:, 3:4], color="tab:purple", label="Antarct Ice Sheet $\psi_2$ [as]")
    plt.legend()
    plt.xlabel('t[-1,+1]')
    plt.title('Units in [as]')
    plt.show()
    x_data = x[:]
    y_data = y[:]
x_data = x[:]
y_data = y[:]
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## transfer the data to the GPU, if you have any
batch_size = time_stamp.shape[0]
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(x_data.to(default_device), y_data.to(default_device))
trainloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
########################################################################################################################
########################################################################################################################
## set the random seed to be able to reproduce the results
torch.manual_seed(123)
########################################################################################################################
########################################################################################################################
## now define the neural networks that we want to train, for different geophysical processes
model_xp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for xp
model_yp = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for yp

model_Bary1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 1
model_Bary2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for barystatic 2

model_GIA1 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL13 for xp yp
model_GIA2 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL23 for xp yp
model_GIA3 = FCN(x.shape[1], 1, 32, 6).to(default_device)  # GIA DL33 for lod

model_Ster1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 1
model_Ster2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for steric 2

model_eq1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 1
model_eq2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## for earthquakes 2

model_icrv1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 1
model_icrv2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 2
model_icrv3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core rotation vector 3

model_lod = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod
model_Bary3Ster3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## model for lod barystatic and steric excitations

model_cmb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 1
model_cmb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque equatorial 2
model_cmb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## core mantle boundary torque axial 3

model_icb1 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 1
model_icb2 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque equatorial 2
model_icb3 = FCN(x_data.shape[1], 1, 32, 6).to(default_device)  ## inner core torque axial 3
########################################################################################################################
########################################################################################################################
## define the optimizer: best is LBFGS
OPT = "LBFGS"
if OPT == "Adam":
    optimizer = torch.optim.Adam(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2)
elif OPT == "AdaBelief":
    optimizer = AdaBelief(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-2,
        eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
elif OPT == "LBFGS":
    optimizer = torch.optim.LBFGS(
        (*model_xp.parameters(), *model_yp.parameters(), *model_Bary1.parameters(), *model_Bary2.parameters(),
         *model_GIA1.parameters(), *model_GIA2.parameters(), *model_GIA3.parameters(),
         *model_Ster1.parameters(), *model_Ster2.parameters(), *model_eq1.parameters(), *model_eq2.parameters(),
         *model_icrv1.parameters(), *model_icrv2.parameters(), *model_icrv3.parameters(), *model_lod.parameters(),
         *model_Bary3Ster3.parameters(), *model_cmb1.parameters(), *model_cmb2.parameters(), *model_cmb3.parameters(),
         *model_icb1.parameters(), *model_icb2.parameters(),
         *model_icb3.parameters()), lr=1e-1,
        max_iter=20)
########################################################################################################################
########################################################################################################################
## define which geophysical models to include

ugiamc = True * 1.
uAIS = True * 1.
uGrIS = True * 1.
uglac = True * 1.
uTWS = True * 1.
usteric = True * 1.
ueq = True * 1.
ugm = True * 1.
ulod = True * 1.
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## define properties of the net, files, etc

files = []

x_physics = x[:].to(default_device)

x_physics = x_physics.requires_grad_(True)

S = 2 * np.pi / 433
w_xp = 1e0
w_yp = 1e0
w_lod = 1e-3
w_gia = 1e-3
w_bary = 1e-3
w_ster = 1e-3
w_eq = 1e-3
w_gm = 1e-10
w_gm_geophys = 1e-3

a_gia = 0.358228557233626
b_gia = 0.00168405708147921
c_gia = 0.000767463763116742
d_gia = 0.352366964476222
e_gia = -6.3775577018066e-03

which_analysis_type = "without_gm"
if which_analysis_type == "with_gm":
    coeff1 = 0.05  # relative importance of xp
    coeff2 = 0.01  # relative importance of yp
elif which_analysis_type == "without_gm":
    coeff1 = 0.003  # relative importance of xp
    coeff2 = 0.0009  # relative importance of yp


########################################################################################################################
########################################################################################################################
## guide to the columns of y:
# column 0:  xp
# column 1:  yp
# column 2: AIS_1
# column 3: AIS_2
# column 4: GrIS_1
# column 5: GrIS_2
# column 6: glac_1
# column 7: glac_2
# column 8: TWS_1
# column 9: TWS_2
# column 10: total_1
# column 11: total_2
# column 12: steric_1
# column 13: steric_2
# column 14: earthquakes 1
# column 15: earthquakes 2
# column 16: CMB 1
# column 17: CMB 2
# column 18: ICB 1
# column 19: ICB 2
# column 20: lod
# column 21: AIS_3
# column 22: GrIS_3
# column 23: glac_3
# column 24: TWS_3
# column 25: total_3
# column 26: steric_3
# column 27: CMB 3
# column 28: ICB 3
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
## train and evaluate

def closure(trainloader=trainloader):
    for x1, y1 in trainloader:
        optimizer.zero_grad()
        ## compute models
        ################################################################################################################
        M_1 = model_xp(x1)
        M_2 = model_yp(x1)
        M_3 = model_Bary1(x1)
        M_4 = model_Bary2(x1)
        M_5 = model_GIA1(x1)
        M_6 = model_GIA2(x1)
        M_7 = model_Ster1(x1)
        M_8 = model_Ster2(x1)
        M_9 = model_eq1(x1)
        M_10 = model_eq2(x1)
        M_11 = model_icrv1(x_physics)  # this only appears in the physical constraints (no data)
        M_12 = model_icrv2(x_physics)  # this only appears in the physical constraints (no data)
        M_13 = model_icrv3(x_physics)  # this only appears in the physical constraints (no data)
        M_14 = model_lod(x1)
        M_15 = model_Bary3Ster3(x1)
        M_16 = model_GIA3(x1)
        M_17 = model_cmb1(x1)
        M_18 = model_cmb2(x1)
        M_19 = model_cmb3(x1)
        M_20 = model_icb1(x1)
        M_21 = model_icb2(x1)
        M_22 = model_icb3(x1)
        ################################################################################################################
        ## compute losses
        tmp_bary1 = uAIS * y1[:, 2:3] + uGrIS * y1[:, 4:5] + uglac * y1[:, 6:7] + uTWS * y1[:, 8:9] + ueq * M_9[:, 0:1]
        tmp_bary2 = uAIS * y1[:, 3:4] + uGrIS * y1[:, 5:6] + uglac * y1[:, 7:8] + uTWS * y1[:, 9:10] + ueq * M_10[:,
                                                                                                             0:1]
        tmp_ster1 = usteric * y1[:, 12:13]
        tmp_ster2 = usteric * y1[:, 13:14]
        tmp_gia1 = Tensor([-1.80465730724889e+31 / C_A]).resize(1, 1)
        tmp_gia2 = Tensor([1.22576269877591e+32 / C_A]).resize(1, 1)
        tmp_gia3 = Tensor([7.07734755270855e+31 / C_earth]).resize(1, 1)
        tmp_baryster3 = uAIS * y1[:, 21:22] + uGrIS * y1[:, 22:23] + uglac * y1[:, 23:24] + uTWS * y1[:, 24:25] + \
                        usteric * y1[:, 26: 27]
        loss_xp = torch.mean((M_1[up_to:, 0:1] - y1[up_to:, 0:1].to(default_device)) ** 2) + \
                  coeff1 * torch.mean((M_1[:up_to, 0:1] - y1[:up_to, 0:1].to(default_device)) ** 2)
        loss_yp = torch.mean((M_2[up_to:, 0:1] - y1[up_to:, 1:2].to(default_device)) ** 2) + \
                  coeff2 * torch.mean((M_2[:up_to, 0:1] - y1[:up_to, 1:2].to(default_device)) ** 2)
        loss_bary1 = torch.mean((M_3[up_to:, 0:1] - tmp_bary1[up_to:, 0:1].to(default_device)) ** 2)
        loss_bary2 = torch.mean((M_4[up_to:, 0:1] - tmp_bary2[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster1 = torch.mean((M_7[up_to:, 0:1] - tmp_ster1[up_to:, 0:1].to(default_device)) ** 2)
        loss_ster2 = torch.mean((M_8[up_to:, 0:1] - tmp_ster2[up_to:, 0:1].to(default_device)) ** 2)
        loss_eq1 = torch.mean((M_9[up_to:, 0:1] - y1[up_to:, 14:15].to(default_device)) ** 2)
        loss_eq2 = torch.mean((M_10[up_to:, 0:1] - y1[up_to:, 15:16].to(default_device)) ** 2)
        loss_lod = torch.mean((M_14[up_to:, 0:1] - y1[up_to:, 20:21].to(default_device)) ** 2)
        loss_bary3 = torch.mean((M_15[up_to:, 0:1] - tmp_baryster3[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia1 = torch.mean((M_5[up_to:, 0:1] - tmp_gia1[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia2 = torch.mean((M_6[up_to:, 0:1] - tmp_gia2[up_to:, 0:1].to(default_device)) ** 2)
        loss_gia3 = torch.mean((M_16[up_to:, 0:1] - tmp_gia3[up_to:, 0:1].to(default_device)) ** 2)
        loss_cmb1 = torch.mean((M_17[up_to:, 0:1] - y1[up_to:, 16:17].to(default_device)) ** 2)
        loss_cmb2 = torch.mean((M_18[up_to:, 0:1] - y1[up_to:, 17:18].to(default_device)) ** 2)
        loss_cmb3 = torch.mean((M_19[up_to:, 0:1] - y1[up_to:, 27:28].to(default_device)) ** 2)
        loss_icb1 = torch.mean((M_20[up_to:, 0:1] - y1[up_to:, 18:19].to(default_device)) ** 2)
        loss_icb2 = torch.mean((M_21[up_to:, 0:1] - y1[up_to:, 19:20].to(default_device)) ** 2)
        loss_icb3 = torch.mean((M_22[up_to:, 0:1] - y1[up_to:, 28:29].to(default_device)) ** 2)

        ## apply physical conditions now
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # 1: Liouville equation with/without solid earth deformation for Barystatic & steric respectively
        xp_geophys = model_xp(x_physics)
        yp_geophys = model_yp(x_physics)
        dxp_geophys = torch.autograd.grad(xp_geophys, x_physics, torch.ones_like(xp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        dyp_geophys = torch.autograd.grad(yp_geophys, x_physics, torch.ones_like(yp_geophys), create_graph=True,
                                          allow_unused=True)[0].to(default_device)[:, 0:1]
        bary1_geophys = model_Bary1(x_physics)
        bary2_geophys = model_Bary2(x_physics)
        ster1_geophys = model_Ster1(x_physics)
        ster2_geophys = model_Ster2(x_physics)
        geophys_loss_xp = torch.mean(
            (dxp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary2_geophys + yp_geophys)) ** 2) + \
                          torch.mean(
                              (dxp_geophys - S * (ster2_geophys + yp_geophys)) ** 2)

        geophys_loss_yp = torch.mean(
            (dyp_geophys - (ks / (ks - k2)) * (1 + k2prime) * S * (bary1_geophys - xp_geophys)) ** 2) + \
                          torch.mean(
                              (dyp_geophys - S * (ster1_geophys - xp_geophys)) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 2: lod Barystatic and steric impacts
        lod_geophys = model_lod(x_physics)
        dlod_geophys = torch.autograd.grad(lod_geophys, x_physics, torch.ones_like(lod_geophys), create_graph=True,
                                           allow_unused=True)[0].to(default_device)[:, 0:1]
        bary3_geophys = model_Bary3Ster3(x_physics)
        dbary3_geophys = \
            torch.autograd.grad(bary3_geophys, x_physics, torch.ones_like(bary3_geophys), create_graph=True,
                                allow_unused=True)[0].to(default_device)[:, 0:1]
        geophys_loss_lod = torch.mean(
            (dlod_geophys - (1 + k2prime) / (1 + 4 / 3 * (C_A / C_earth) * (k2 / ks)) * dbary3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 3: GIA and mantle convection models

        gia1_geophys = model_GIA1(x_physics)
        gia2_geophys = model_GIA2(x_physics)
        gia3_geophys = model_GIA3(x_physics)

        geophys_loss_gia1 = torch.mean((a_gia * xp_geophys + b_gia * yp_geophys - c_gia * gia1_geophys) ** 2)
        geophys_loss_gia2 = torch.mean((b_gia * xp_geophys + d_gia * yp_geophys - c_gia * gia2_geophys) ** 2)
        geophys_loss_gia3 = torch.mean((lod_geophys - e_gia - gia3_geophys) ** 2)
        ################################################################################################################
        ################################################################################################################
        # 4: Geomagnetism
        cmb1_geophys = model_cmb1(x_physics)
        cmb2_geophys = model_cmb2(x_physics)
        cmb3_geophys = model_cmb3(x_physics)
        icb1_geophys = model_icb1(x_physics)
        icb2_geophys = model_icb2(x_physics)
        icb3_geophys = model_icb3(x_physics)

        dicrv1_geophys = torch.autograd.grad(M_11, x_physics, torch.ones_like(M_11), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv2_geophys = torch.autograd.grad(M_12, x_physics, torch.ones_like(M_12), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]
        dicrv3_geophys = torch.autograd.grad(M_13, x_physics, torch.ones_like(M_13), create_graph=True,
                                             allow_unused=True)[0].to(default_device)[:, 0:1]

        geophys_loss_gm1 = torch.mean((I_m11 * dxp_geophys + I_m12 * dyp_geophys + I_m13 * dlod_geophys -
                                       (M_11 - xp_geophys) * GammaXtau - cmb1_geophys) ** 2)
        geophys_loss_gm2 = torch.mean((I_m21 * dxp_geophys + I_m22 * dyp_geophys + I_m23 * dlod_geophys -
                                       (M_12 - yp_geophys) * GammaXtau - cmb2_geophys) ** 2)
        geophys_loss_gm3 = torch.mean((I_m31 * dxp_geophys + I_m32 * dyp_geophys + I_m33 * dlod_geophys -
                                       (M_13 - lod_geophys) * GammaXtau - cmb3_geophys) ** 2)

        geophys_loss_gm4 = torch.mean((I_c11 * dicrv1_geophys + I_c12 * dicrv2_geophys + I_c13 * dicrv3_geophys +
                                       (M_11 - xp_geophys) * GammaXtau - icb1_geophys) ** 2)
        geophys_loss_gm5 = torch.mean((I_c21 * dicrv1_geophys + I_c22 * dicrv2_geophys + I_c23 * dicrv3_geophys +
                                       (M_12 - yp_geophys) * GammaXtau - icb2_geophys) ** 2)
        geophys_loss_gm6 = torch.mean((I_c31 * dicrv1_geophys + I_c32 * dicrv2_geophys + I_c33 * dicrv3_geophys +
                                       (M_13 - lod_geophys) * GammaXtau - icb3_geophys) ** 2)

        ################################################################################################################
        ################################################################################################################
        # add the losses together
        loss = w_xp * loss_xp + w_yp * loss_yp + w_lod * loss_lod + \
               w_bary * (loss_bary1 + loss_bary2 + loss_bary3) + \
               w_eq * (loss_eq1 + loss_eq2) + \
               w_ster * (loss_ster1 + loss_ster2) + \
               w_gia * (loss_gia1 + loss_gia2 + loss_gia3) + w_gm * (
                       loss_cmb1 + loss_cmb2 + loss_cmb3 + loss_icb1 + loss_icb2 + loss_icb3) + \
               w_gm_geophys * (
                       geophys_loss_xp + geophys_loss_yp + geophys_loss_lod + geophys_loss_gia1 + geophys_loss_gia2 +
                       geophys_loss_gia3 + geophys_loss_gm1 + geophys_loss_gm2 + geophys_loss_gm3 + geophys_loss_gm4 +
                       geophys_loss_gm5 + geophys_loss_gm6)

        loss.backward()
        return loss


########################################################################################################################
########################################################################################################################
## train the model

N_epochs = 150  # number of training epochs

for i in range(N_epochs):
    print(f"analysis for the eopch {i + 1}")
    model_xp.train()
    model_yp.train()
    model_Bary1.train()
    model_Bary2.train()
    model_GIA1.train()
    model_GIA2.train()
    model_GIA3.train()
    model_Ster1.train()
    model_Ster2.train()
    model_eq1.train()
    model_eq2.train()
    model_icrv1.train()
    model_icrv2.train()
    model_icrv3.train()
    model_lod.train()
    model_Bary3Ster3.train()
    model_cmb1.train()
    model_cmb2.train()
    model_cmb3.train()
    model_icb1.train()
    model_icb2.train()
    model_icb3.train()

    optimizer.step(closure)

    # plot the result as training progresses
    if (i + 1) % 5 == 0:
        yh_xp1 = model_xp(x.to(default_device)).cpu().detach()
        yh_yp1 = model_yp(x.to(default_device)).cpu().detach()

        plt.figure()
        plt.subplot(2, 1, 1)
        plot_result(x[:, 0:1], y[:, 0:1], x_data[up_to:, 0:1], y_data[up_to:, 0:1], yh_xp1, plot_title=True,
                    plot_ylabel="IntAnn $x_p$")
        plt.subplot(2, 1, 2)
        plot_result(x[:, 0:1], y[:, 1:2], x_data[up_to:, 0:1], y_data[up_to:, 1:2], yh_yp1, plot_ylabel="IntAnn $y_p$",
                    plot_xlabel=True)

        file = "figures/pinn_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        plt.close()
    else:
        plt.close("all")
########################################################################################################################
########################################################################################################################
## save the GIF animation file
save_folder_name = "results_PINNs"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

save_gif_PIL(os.path.join(cwd, save_folder_name, f"pinn_{which_analysis_type}_{OPT}.gif"), files, fps=10, loop=2)
########################################################################################################################
########################################################################################################################
## evaluate the model
model_xp.eval()
model_yp.eval()
model_Bary1.eval()
model_Bary2.eval()
model_GIA1.eval()
model_GIA2.eval()
model_GIA3.eval()
model_Ster1.eval()
model_Ster2.eval()
model_eq1.eval()
model_eq2.eval()
model_icrv1.eval()
model_icrv2.eval()
model_icrv3.eval()
model_lod.eval()
model_Bary3Ster3.eval()
model_cmb1.eval()
model_cmb2.eval()
model_cmb3.eval()
model_icb1.eval()
model_icb2.eval()
model_icb3.eval()

xp1 = model_xp(x[:up_to].to(default_device)).cpu().detach()
yp1 = model_yp(x[:up_to].to(default_device)).cpu().detach()
E_xp1 = xp1 - y[:up_to, 0:1]
E_yp1 = yp1 - y[:up_to, 1:2]
########################################################################################################################
########################################################################################################################
## save the results

np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_xp.txt"), E_xp1)
np.savetxt(os.path.join(cwd, save_folder_name, f"errors_{which_analysis_type}_yp.txt"), E_yp1)


