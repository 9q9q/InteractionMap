"""Binary to run 2L SC on GPU."""
# TODO: add flags
# TODO: refactor and make readable
# TODO: figure out multi GPU?
# TODO: implement resize convolution to get rid of checker https://distill.pub/2016/deconv-checkerboard/

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage import io
from tensorboardX import SummaryWriter
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from SDPC_PCB.Network import LayerPC, Network
from SDPC_PCB.Coding import ML_Lasso, ML_FISTA
from SDPC_PCB.DataTools import gaussian_kernel, norm, to_cuda
from SDPC_PCB.Monitor import Monitor
from SDPC_PCB.Optimizers import mySGD, myAdam


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    return plt


def reconstruction(Net, gamma):
    reco = [None] * (Net.nb_layers)
    for i in range(Net.nb_layers-1, -1, -1):
        reco[i] = gamma[i]
        for j in range(i, -1, -1):
            reco[i] = Net.layers[j].backward(reco[i])
    return reco


class MyDataset(Dataset):
    """Dataset created by images from specified folder."""

    def __init__(self, root_dir, patch_size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            patch_size (int): Length of one side of square patch.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(glob.glob(os.path.join(self.root_dir, "*.png")))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "im_{}.png".format(idx))
        image = torch.Tensor(io.imread(img_name)).float()
        image = torch.permute(image, (2, 0, 1))
        image = image[0, :, :].unsqueeze(0)  # only one channel
        assert image.shape[-1] == self.patch_size

        if self.transform:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-bs", default=256)
    # parser.add_argument("-data_path", default="pacmen1024_64x64")
    # parser.add_argument("-ps", default=64)
    # parser.add_argument("-dict_sizes", default=[(16, 1, 8, 8), (32, 16, 8, 8)])
    # parser.add_argument("-lr", default=[1e-3, 5e-2])
    # parser.add_argument("-epochs", default=100)
    # parser.add_argument("-infer_iters", default=1000)
    # parser.add_argument("-name")
    # args = parser.parse_args()

    batch_size = 128
    dataset_path = "pacmen1024_64x64"
    patch_size = 64
    # [Layer 1, Layer 2] [num_channels, in_channels, H, W]
    dict_sizes = [(16, 1, 16, 16), (32, 16, 16, 16)]
    # dictionaries learning rate [Layer1, Layer2]  # original params
    # l_r = [1e-4, 5e-3]  # these don't work for smaller patches, get NaN
    l_r = [1e-1, 5e-1]  # dictionaries learning rate [Layer1, Layer2]
    mask = False

    l_rv = [1e-3, 1e-3]  # normalizer learning rate [Layer1, Layer2]
    l = [0.4, 1.6]  # Sparsity parameters [Layer1, Layer2]
    b = 1  # Feedback strength parameter. b=0 --> Hila, b=1 --> 2L-SPC
    v_i = [10, 10]  # Initial normalizer value [Layer1, Layer2]
    nb_epoch = 500  # number of training epochs
    infer_iters = 1000  # max inference iters
    Use_tb = True  # Use to activate tensorboard monitoring
    save = True  # Use to run the entire simulation : TAKE HOURS. Use False to load previous simulation
    log_dir = "logs"
    model_dir = "models"
    # model_name = "{}_bs{}_ep{}_infer{}_6".format(
    #     dataset_path, batch_size, nb_epoch, infer_iters, patch_size)
    model_name = "big_lr"
    params = {"batch_size": batch_size, "dataset_path": dataset_path,
              "patch_size": patch_size, "dict0_size": dict_sizes[0],
              "dict1_size": dict_sizes[1], "lr0": l_r[0], "lr1": l_r[1],
              "mask": mask, "epochs": nb_epoch, "infer_iters": infer_iters,
              "model_name": model_name}
    param_dict = torch.nn.ParameterDict(
        parameters=params)

    os.makedirs(os.path.join(log_dir, model_name), exist_ok=False)
    with open(os.path.join(log_dir, model_name, "params.txt"), "w") as f:
        f.write(str(list(param_dict.items())))

    # Load data.
    model_path = os.path.join(model_dir, model_name+".pkl")
    transform = to_cuda()
    dataset = MyDataset(dataset_path, patch_size=patch_size,
                        transform=transform)
    database = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, drop_last=True)
    input_size = list(next(iter(database)).shape)
    print("data batch size: {}".format(input_size))

    # Gaussian masks for the dictionaries
    mask_g = [gaussian_kernel(dict_sizes[0], sigma=30), gaussian_kernel(
        dict_sizes[1], sigma=30)]  # not sure why do this

    # Definition of the layers, network and sparse coding algorithm
    layer = [LayerPC(dict_sizes[0], stride=2, b=b, v=v_i[0], v_size=dict_sizes[0][0], out_pad=0),
             LayerPC(dict_sizes[1], stride=1, b=b, v=v_i[1], v_size=dict_sizes[0][1], out_pad=0)]

    Net = Network(layer, input_size=input_size)
    Loss = ML_Lasso(Net, [l[0], l[1]])
    Pursuit = ML_FISTA(Net, Loss, max_iter=infer_iters, th=1e-4, mode='eigen')

    # Optimizer initialization
    opt_dico = [None] * (Net.nb_layers + 1)
    for i in range(0, Net.nb_layers):
        opt_dico[i] = mySGD([{'params': Net.layers[i].dico}],
                            lr=l_r[i], momentum=0.9, normalize=True)

    opt_v = [myAdam([{'params': Net.layers[i].v}], lr=l_rv[i], normalize=False)
             for i in range(Net.nb_layers)]

    L = [None] * (Net.nb_layers)
    L_v = [None] * (Net.nb_layers)
    reco = [None] * (Net.nb_layers)

    if Use_tb:
        nrows = [8, 8, 8, 8, 8, 8, 8]
        writer = SummaryWriter(os.path.join(log_dir, model_name))
        M = Monitor(Net, writer, n_row=nrows)
        writer.add_hparams(params, metric_dict={})

    l2_loss = torch.zeros(2, nb_epoch*len(database))
    l1_loss = torch.zeros(2, nb_epoch*len(database))
    for e in tqdm(range(nb_epoch)):
        for idx_batch, data in enumerate(database):
            if torch.cuda.is_available():
                batch = data.cuda()
            else:
                batch = data
            gamma, it, Loss_G, delta = Pursuit.coding(batch)

            for i in range(Net.nb_layers):
                Net.layers[i].dico.requires_grad = True
                L[i] = Loss.F(batch, gamma, i, do_feedback=False).div(
                    batch.size()[0])  # Unsupervised
                L[i].backward()
                Net.layers[i].dico.requires_grad = False
                opt_dico[i].step()
                opt_dico[i].zero_grad()

                # Mask
                if mask:
                    Net.layers[i].dico *= mask_g[i]
                Net.layers[i].dico /= norm(Net.layers[i].dico)

                l2_loss[i, idx_batch] = L[i].detach()
                l1_loss[i, idx_batch] = gamma[i].detach(
                ).sum().div(gamma[i].size(0))

            for i in range(Net.nb_layers):
                Net.layers[i].v.requires_grad = True  # turn_on(i)
                L_v[i] = Loss.F_v(batch, gamma, i).div(batch.size()[0])
                L_v[i].backward()
                Net.layers[i].v.requires_grad = False  # turn_off(i)
                opt_v[i].step()
                opt_v[i].zero_grad()

            if Use_tb:
                if (idx_batch % 10) == 0:
                    writer.add_scalar('FISTA_iterations', it, idx_batch)
                    M.MonitorGamma(gamma, idx_batch, option=[
                                   'NNZ', '%', 'Sum', 'V'])
                    M.MonitorList(L, 'Loss_Dico', idx_batch)
                    M.MonitorList(L_v, 'Loss_v', idx_batch)
                    M.MonitorDicoBP(idx_batch)
                    M.ComputeHisto(gamma)

                if (idx_batch % 10) == 0:
                    reco = [None] * (Net.nb_layers)
                    for i in range(Net.nb_layers-1, -1, -1):
                        reco[i] = gamma[i]
                        for j in range(i, -1, -1):
                            reco[i] = Net.layers[j].backward(reco[i])
                        reco_image = make_grid(
                            reco[i], normalize=True, pad_value=1)
                        writer.add_image(
                            'Reco/L{0}'.format(i), reco_image, idx_batch)

    output_exp = {'Net': Net,
                  'Loss': Loss,
                  'Pursuit': Pursuit,
                  'l2_loss': l2_loss,
                  'l1_loss': l1_loss
                  }
    with open(model_path, 'wb') as file:
        pickle.dump(output_exp, file, pickle.HIGHEST_PROTOCOL)
