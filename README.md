# text2art
Hackamatics project 2021 with the aim to pull together all text2art resources and get it running on our GPUs.


## Getting started:
```
pip3 install virtualenv 
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
make deps
git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers
```