# Text2Art
This project was completed during a Speechmatics Hackathon in December 2021 with the goal of pulling together various text-to-art resources from different Google Colab notebooks and testing their to a variety of text prompts. Scripts were generated to make generation in many different styles very quick as well as style transfer on existing images. Two generative models were used for comparison: VQGAN and DALLE. The output of these models can be improved with ESRGAN upscaling as a post processing step. The models can be submitted to a GPU queue for efficient parallelisation rather than relying on Colab instances. Our team won first prize!

## Getting Started
These instructions will setup the environment and download the model:

```
make env
source venv/bin/activate
make deps
make models
```

To run the image generation given a text prompt, run the following in a Speechmatics dev-vm:
```
# grab GPU then run
qlogin -now n -pe smp 1 -q aml-gpu.q -l gpu=1 -pty y -N D_$(whoami)
cd ~/git/text2art && source venv/bin/activate
python3 -m text2art.vqgan.run --prompts "Three engineers hard at work during Hackamatics #artstation"
```

## Example Outputs

Update (now in January 2023): these were generated just as the text to art craze started back at the end of 2021. It is amazing how far the technology has come since then with Stable Diffusion models instead of VQGAN and the emergence of tools such as Mid Journey which completely blow me away.

### VQGAN & Different Styles
<img width="1190" alt="Screenshot 2023-01-03 at 23 14 22" src="https://user-images.githubusercontent.com/25829615/210457015-6cebf91a-cddb-4f0f-a29f-cc490b363420.png">
<img width="1194" alt="Screenshot 2023-01-03 at 23 14 30" src="https://user-images.githubusercontent.com/25829615/210457031-cfc57181-273b-409f-9fef-72b245f91710.png">

### Style Transfer on the Speechmatics Stand
<img width="1200" alt="Screenshot 2023-01-03 at 23 18 10" src="https://user-images.githubusercontent.com/25829615/210457250-f12ad7c4-aa9c-4146-86a6-1fd788c71f4f.png">

### DALLE & Food Trucks!
<img width="1167" alt="Screenshot 2023-01-03 at 23 15 07" src="https://user-images.githubusercontent.com/25829615/210457077-84793022-528f-439a-bf77-671cc97139d9.png">
<img width="1171" alt="Screenshot 2023-01-03 at 23 15 34" src="https://user-images.githubusercontent.com/25829615/210457119-3056db18-74ee-4eb9-8a51-359404a5f33d.png">


## Acknowledgments
[Speechmatics](https://www.speechmatics.com) for hosting the hackathon and providing the resources (and prizes!) for this project.

