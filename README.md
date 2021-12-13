# text2art
Hackamatics project 2021 with the aim to pull together all text2art resources and get it running on our GPUs.


## Initial setup
```
# Setup environment and download models (one time only)
make env
make activate
make deps
make models
```

## Run text2art
```
# grab GPU then run
qlogin -now n -pe smp 1 -q aml-gpu.q -l gpu=1 -pty y -N D_$(whoami)
cd ~/git/text2art && make activate
python3 -m text2art.vqgan.run --prompts "Three engineers hard at work during Hackamatics #artstation"
```