import argparse
import os
from PIL import Image
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from time import perf_counter

from clip import clip
from text2art.vqgan.util import (
    resample,
    fetch,
    parse_prompt,
    load_vqgan_model,
    resize_image,
    replace_grad,
    clamp_with_grad,
    urlify,
)


class MakeCutouts(nn.Module):
    """Take an image and return number_of_cuts number of patches which are of size cut_size.
    We use this function to return multiple patches of the same image but of a cut
    size that CLIP is expecting."""

    def __init__(self, cut_size, number_of_cuts, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.number_of_cuts = number_of_cuts
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.number_of_cuts):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


class Prompt(nn.Module):
    """Initialise your prompt used to drive the encoded output towards. The forward
    # method calculates the loss (distance between embeddings)"""

    def __init__(self, embed, weight=1.0, stop=float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        # Calculate distance metric between embedding and input
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def vector_quantize(x, codebook):
    """Quantise a latent variable to the closest vector in the codebook"""
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


def synthesize(z, model):
    """Quantise the embeddings and pass through the decoder to generate an image"""
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run VQGAN and CLIP system on given text prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts", nargs="+", default=[], required=True)
    parser.add_argument("--image_dir", type=str, default="./images")
    parser.add_argument("--image_prompts", nargs="+", default=[])
    parser.add_argument("--noise_prompt_seeds", nargs="+", default=[])
    parser.add_argument("--noise_prompt_weights", nargs="+", default=[])
    parser.add_argument("--size", nargs="+", default=[480, 480])
    parser.add_argument("--init_image", type=str)
    parser.add_argument("--init_weight", type=float, default=0.0)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument(
        "--vqgan_config", type=str, default="models/VQGAN/vqgan_imagenet_f16_1024.yaml"
    )
    parser.add_argument(
        "--vqgan_checkpoint", type=str, default="models/VQGAN/vqgan_imagenet_f16_1024.ckpt"
    )
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--number_of_cuts", type=int, default=64)
    parser.add_argument("--cut_pow", type=float, default=1.0)
    parser.add_argument("--display_freq", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.size = [int(x) for x in args.size]

    # General setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create output dir for images based on text prompt
    text_prompt = urlify(args.prompts[0])
    os.makedirs(args.image_dir, exist_ok=True)

    # Load pre trained models
    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    # Set the cut size of the images based on the CLIP system.
    cut_size = perceptor.visual.input_resolution
    make_cutouts = MakeCutouts(cut_size, args.number_of_cuts, cut_pow=args.cut_pow)
    # Normalise function to be used on these image "cutouts" in order to get an average
    # image that represents the resolution required.
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )

    # Save various dimensions from VQGAN
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    f = 2 ** (model.decoder.num_resolutions - 1)
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    # Initialise our parameters to train z, which are the internal embeddings of the VQGAN
    # embeddings are learnt that lead to a decoded image that best matches the text embedding via CLIP
    if args.init_image:
        # If initialisation image provided, encode to get the embeddings as a starting point
        pil_image = Image.open(fetch(args.init_image)).convert("RGB")
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        # Choose random embeddings - decoding this looks like noise
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    model_params = sum(p.numel() for p in model.parameters()) / 1.0e6
    perceptor_params = sum(p.numel() for p in perceptor.parameters()) / 1.0e6
    print(f"VQGAN: e_dim={e_dim}, f={f}, n_toks={n_toks}, params={model_params:.2f}M")
    print(f"CLIP: resolution={cut_size}, params={perceptor_params:.2f}M")
    z_params = sum(z.shape)
    print(f"z: params={z_params}")

    # Compile a list of prompts made up of embeddings to drive the encoded output image towards
    prompt_list = []
    # 1) Prompt via text to drive output towards
    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        prompt_list.append(Prompt(embed, weight, stop).to(device))

    # 2) Prompt via image to drive output towards
    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(fetch(path)).convert("RGB"), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        prompt_list.append(Prompt(embed, weight, stop).to(device))

    # 3) Random prompts based on embedings sampled from the CLIP model
    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        prompt_list.append(Prompt(embed, weight).to(device))

    step = 0
    step_times = []
    while True:
        step_start_time = perf_counter()
        opt.zero_grad()

        # Get output image based on our current embeddings z (that we are training)
        out = synthesize(z, model)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        losses = []

        # Incentivise loss to keep output image close to initial given image by penalising
        # z if it becomes far away from z_orig. Used in conjuncion with init_image.
        if args.init_weight:
            losses.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

        # Get loss for each provided prompt to drive closer to provided text or image embeddings
        for prompt in prompt_list:
            losses.append(prompt(iii))

        # Log losses and save out progress images
        if step % args.display_freq == 0:
            if len(step_times) > 0:
                av_step_time = sum(step_times) / len(step_times)
            else:
                av_step_time = 0
            print(
                f"step: {step}, loss: {sum(losses).item():.5f}, time: {av_step_time:.4f}",
                flush=True,
            )
            if len(losses) > 1:
                losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
                print(f"losses: {losses_str}", flush=True)
            with torch.no_grad():
                out = synthesize(z, model)
                TF.to_pil_image(out[0].cpu()).save(f"{args.image_dir}/{text_prompt}_step{i}.png")

        # Sum losses for each prompt and update parameters accordingly
        loss = sum(losses)
        loss.backward()
        opt.step()

        # Pin our embeddings z to within values expected by VQGAN
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

        if args.steps != -1 and step >= args.steps:
            with torch.no_grad():
                out = synthesize(z, model)
                TF.to_pil_image(out[0].cpu()).save(f"{args.image_dir}/final.png")
            break

        step += 1
        elapsed = perf_counter() - step_start_time
        step_times.append(elapsed)


if __name__ == "__main__":
    main()
