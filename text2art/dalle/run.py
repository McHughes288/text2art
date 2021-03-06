from clip import model
import torch
import torchvision
import torchvision.transforms.functional as TF
import argparse

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import clip
import os
from dall_e import unmap_pixels, load_model

from text2art.vqgan.util import urlify


class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        self.normu = torch.nn.Parameter(torch.randn(1, 8192, 64, 64).cuda())

    def forward(self):
        normu = torch.nn.functional.gumbel_softmax(self.normu.view(1, 8192, -1), dim=-1).view(
            1, 8192, 64, 64
        )
        return normu


def main():
    parser = argparse.ArgumentParser(
        description="Run DALL-E and CLIP system on given text prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts", nargs="+", default=[], required=True)
    parser.add_argument("--image_dir", type=str, default="./images")
    parser.add_argument("--size", nargs="+", default=[480, 480])
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--number_of_cuts", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    args.size = [int(x) for x in args.size]
    sideX, sideY = args.size

    os.makedirs(args.image_dir, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load CLIP and DALL-E models
    perceptor = clip.load(args.clip_model, jit=True)[0].eval()
    model = load_model("https://cdn.openai.com/dall-e/decoder.pkl", "cuda")

    lats = Pars().cuda()
    mapper = [lats.normu]
    optimizer = torch.optim.Adam([{"params": mapper, "lr": args.lr}])
    eps = 0

    # Tokenize text prompt and encode in CLIP
    tx = clip.tokenize(args.prompts)
    t = perceptor.encode_text(tx.cuda()).detach().clone()

    # Normalise function to be used on these image "cutouts" in order to get an average
    # image that represents the resolution required.
    nom = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    i = 0
    while True:
        out = unmap_pixels(torch.sigmoid(model(lats())[:, :3].float()))
        cutn = args.number_of_cuts  # improves quality
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
            offsety = torch.randint(0, sideY - size, ())
            apper = out[:, :, offsetx : offsetx + size, offsety : offsety + size]
            apper = torch.nn.functional.interpolate(apper, (224, 224), mode="bilinear")
            p_s.append(apper)
        into = torch.cat(p_s, 0)
        into = nom(into)
        iii = perceptor.encode_image(into)
        lat_l = 0
        loss1 = [lat_l, 10 * -torch.cosine_similarity(t, iii).view(-1, 1).T.mean(1)]
        loss = loss1[0] + loss1[1]
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_every == 0:
            with torch.no_grad():
                al = unmap_pixels(torch.sigmoid(model(lats())[:, :3]).cpu().float()).numpy()
            TF.to_pil_image(torch.tensor(al[0]).cpu()).save(
                f"{args.image_dir}/{urlify(args.prompts[0])}_step{i}.png"
            )

        if i > args.steps:
            TF.to_pil_image(out[0].cpu()).save(f"{args.image_dir}/final.png")
            break

        i += 1


if __name__ == "__main__":
    main()
