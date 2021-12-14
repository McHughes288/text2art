from clip import model
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import PIL
import matplotlib.pyplot as plt

import os
import random
from IPython import display
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import clip

import io
import sys
import requests

from dall_e import map_pixels, unmap_pixels, load_model


def displ(img, pre_scaled=True):
    img = np.array(img)[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    # if not pre_scaled:
    # img = scale(img, 48*4, 32*4)
    # imageio.imwrite(str(3) + '.png', np.array(img))
    return display.Image(str(3) + ".png")


def gallery(array, ncols=2):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (
        array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, intensity)
    )
    return result


def card_padded(im, to_pad=3):
    return np.pad(
        np.pad(
            np.pad(im, [[1, 1], [1, 1], [0, 0]], constant_values=0),
            [[2, 2], [2, 2], [0, 0]],
            constant_values=1,
        ),
        [[to_pad, to_pad], [to_pad, to_pad], [0, 0]],
        constant_values=0,
    )


def get_all(img):
    img = np.transpose(img, (0, 2, 3, 1))
    cards = np.zeros((img.shape[0], sideX + 12, sideY + 12, 3))
    for i in range(len(img)):
        cards[i] = card_padded(img[i])
    print(img.shape)
    cards = gallery(cards)
    # imageio.imwrite(str(3) + '.png', np.array(cards))
    return display.Image(str(3) + ".png")


def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f"min dim for image {s} < {target_image_size}")

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        self.normu = torch.nn.Parameter(torch.randn(1, 8192, 64, 64).cuda())

    def forward(self):
        normu = torch.nn.functional.gumbel_softmax(self.normu.view(1, 8192, -1), dim=-1).view(
            1, 8192, 64, 64
        )
        return normu


def checkin(loss):
    print(itt, "loss", loss[1].item())

    with torch.no_grad():
        al = unmap_pixels(torch.sigmoid(model(lats())[:, :3]).cpu().float()).numpy()
    for allls in al:
        displ(allls)
        # display.display(display.Image(str(3)+'.png'))
        # print('\n')
    # the people spoke and they love "ding"
    # output.eval_js('new Audio("https://freesound.org/data/previews/80/80921_1022651-lq.ogg").play()')


def ascend_txt():
    out = unmap_pixels(torch.sigmoid(model(lats())[:, :3].float()))

    cutn = 64  # improves quality
    p_s = []
    for ch in range(cutn):
        size = int(
            sideX
            * torch.zeros(
                1,
            )
            .normal_(mean=0.8, std=0.3)
            .clip(0.5, 0.98)
        )
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = out[:, :, offsetx : offsetx + size, offsety : offsety + size]
        apper = torch.nn.functional.interpolate(apper, (224, 224), mode="bilinear")
        p_s.append(apper)
    into = torch.cat(p_s, 0)
    # into = torch.nn.functional.interpolate(out, (224,224), mode='nearest')

    into = nom(into)

    iii = perceptor.encode_image(into)

    llls = lats()
    lat_l = 0

    return [lat_l, 10 * -torch.cosine_similarity(t, iii).view(-1, 1).T.mean(1)]


def train(i):
    loss1 = ascend_txt()
    loss = loss1[0] + loss1[1]
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if itt % 100 == 0:
        checkin(loss1)


clip.available_models()

perceptor, preprocess = clip.load("ViT-B/32", jit=True)
perceptor = perceptor.eval()


prompt = "three engineers hard at work"

im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape


target_image_size = sideX
model = load_model("https://cdn.openai.com/dall-e/decoder.pkl", "cuda")


lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{"params": mapper, "lr": 0.1}])
eps = 0

tx = clip.tokenize(prompt)
t = perceptor.encode_text(tx.cuda()).detach().clone()

nom = torchvision.transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)

with torch.no_grad():
    mult = 1
    al = unmap_pixels(torch.sigmoid(model(lats()).cpu().float())).numpy()
    for allls in al:
        displ(allls[:3])
        print("\n")
    # print(torch.topk(lats().view(1, 8192, -1), k=3, dim=-1))


itt = 0
for asatreat in range(10000):
    train(itt)
    itt += 1
