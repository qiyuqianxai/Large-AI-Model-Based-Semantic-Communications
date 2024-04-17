import json
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
import os
import warnings
import torchvision.datasets as dset
from PIL import Image
warnings.filterwarnings("ignore")
from base_nets import base_net
from channel_nets import channel_net,MutualInfoSystem,sample_batch
from neural_nets import SCNet
import time
import numpy as np
import torchvision
import random
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
torch.cuda.set_device(0)
class params():
    checkpoint_path = "checkpoints"
    device = "cuda"
    dataset = "../datasets/val2017_seg"
    log_path = "logs"
    epoch = 100
    lr = 1e-3
    batchsize = 128
    snr = 25
    weight_delay = 1e-6
    use_ASC = True
    save_model_name = "LAM-SC"

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def show_images(pred_images, filename):
    imgs_sample = (pred_images.data + 1) / 2.0
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

class custom_datasets(Dataset):
    def __init__(self, data):
        self.data = data.imgs
        self.img_transform = self.transform()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        img = Image.open(self.data[item][0]).convert('RGB')
        img = self.img_transform(img)
        return img, self.data[item][0]

    def transform(self):
        compose = [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        return transforms.Compose(compose)

def train_SCNet(model, train_dataloader, arg:params):
    # laod weights
    weights_path = os.path.join(arg.checkpoint_path,f"{arg.save_model_name}_snr{arg.snr}.pth")
    model = model.to(arg.device)
    muInfoNet = MutualInfoSystem()
    muInfoNet.load_state_dict(torch.load(os.path.join(arg.checkpoint_path,"MI.pth"), map_location="cpu"))
    muInfoNet.to(arg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr,
                                             weight_decay=arg.weight_delay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                      factor=0.1, patience=100,
                                                                      verbose=True, threshold=0.0001,
                                                                      threshold_mode='rel',
                                                                      cooldown=0, min_lr=0, eps=1e-08)
    # define loss function
    mse = nn.MSELoss()
    model.train()
    loss_record = []
    for epoch in range(arg.epoch):
        start = time.time()
        losses = []
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.to(arg.device)
            encoding,encoding_with_noise,decoding = model(x)
            # compute MI loss
            batch_joint = sample_batch(1, 'joint', encoding, encoding_with_noise).to(arg.device)
            batch_marginal = sample_batch(1, 'marginal', encoding, encoding_with_noise).to(arg.device)
            t = muInfoNet(batch_joint)
            et = torch.exp(muInfoNet(batch_marginal))
            loss_MI = torch.mean(t) - torch.log(torch.mean(et)) # or use mse(encoding,encoding_with_noise) for simplicity
            # compute SC loss
            loss_SC = mse(decoding,x)
            loss = loss_MI + loss_SC
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
        losses = np.mean(losses)
        loss_record.append(losses)
        print(f"epoch {epoch} | MI loss: {loss_MI.item()} | SC loss: {loss_SC.item()} | total loss: {loss.item()} | waste time: {time.time() - start}")
        if epoch%5==0:
            os.makedirs(os.path.join(arg.log_path, f"{arg.snr}"),exist_ok=True)
            show_images(x.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}",f"{arg.save_model_name}_imgs.jpg"))
            show_images(decoding.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}",f"{arg.save_model_name}_rec_imgs.jpg"))
        with open(os.path.join(arg.log_path,f"{arg.save_model_name}_snr{arg.snr}_loss.json"),"w",encoding="utf-8")as f:
            f.write(json.dumps(loss_record,indent=4,ensure_ascii=False))
        torch.save(model.state_dict(), weights_path)

def train_MASKNet(model, train_dataloader, arg:params):
    # laod weights
    weights_path = os.path.join(arg.checkpoint_path, f"{arg.save_model_name}_snr{arg.snr}.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    muInfoNet = MutualInfoSystem()
    muInfoNet.load_state_dict(torch.load(os.path.join(arg.checkpoint_path, "MI.pth"), map_location="cpu"))
    muInfoNet.to(arg.device)
    model = model.to(arg.device)
    # Train the MASKNet and frozen the SCNet
    for param in model.parameters():
        param.requires_grad = False
    for param in model.isc_model.Mask.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.isc_model.Mask.parameters(), lr=arg.lr,
                                 weight_decay=arg.weight_delay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=100,
                                                           verbose=True, threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    mse = nn.MSELoss()
    model.train()
    loss_record = []
    for epoch in range(30):
        start = time.time()
        losses = []
        for i, (x, y) in enumerate(train_dataloader):
            model.zero_grad()
            x = x.to(arg.device)
            encoding, encoding_with_noise, decoding = model(x)
            # compute MI loss
            batch_joint = sample_batch(5, 'joint', encoding, encoding_with_noise).to(arg.device)
            batch_marginal = sample_batch(5, 'marginal', encoding, encoding_with_noise).to(arg.device)

            t = muInfoNet(batch_joint)
            et = torch.exp(muInfoNet(batch_marginal))
            loss_MI = torch.mean(t) - torch.log(torch.mean(et))
            # compute SC loss
            loss_SC = mse(decoding, x)
            loss = loss_MI + loss_SC
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
        losses = np.mean(losses)
        loss_record.append(losses)
        print(
            f"epoch {epoch} | MI loss: {loss_MI.item()} | Rec loss: {loss_SC.item()} | total loss: {loss.item()} | waste time: {time.time() - start}")
        if epoch % 5 == 0:
            os.makedirs(os.path.join(arg.log_path, f"{arg.snr}"), exist_ok=True)
            show_images(x.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}", f"{arg.save_model_name}_imgs.jpg"))
            show_images(decoding.detach().cpu(), os.path.join(arg.log_path, f"{arg.snr}", f"{arg.save_model_name}_rec_imgs.jpg"))
        with open(os.path.join(arg.log_path, f"{arg.save_model_name}_snr{arg.snr}_loss.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(loss_record, indent=4, ensure_ascii=False))
        weights_path = os.path.join(arg.checkpoint_path, f"{arg.save_model_name}_snr{arg.snr}.pth")
        torch.save(model.state_dict(), weights_path)


if __name__ == '__main__':
    same_seeds(1024)
    arg = params()
    Img_data = dset.ImageFolder(root=arg.dataset)
    datasets = custom_datasets(Img_data)
    train_dataloader = DataLoader(dataset=datasets, batch_size=arg.batchsize, shuffle=True, num_workers=0,
                                  drop_last=False)
    # training SCNet
    SC_model = SCNet(input_dim=3, ASC=False)
    channel_model = channel_net(in_dims=5408, snr=arg.snr)
    model = base_net(SC_model, channel_model)
    train_SCNet(model, train_dataloader, arg)

    # training MaskNet
    SC_model = SCNet(input_dim=3, ASC=True)
    channel_model = channel_net(in_dims=5408, snr=arg.snr)
    model = base_net(SC_model, channel_model)
    train_MASKNet(model, train_dataloader, arg)

















