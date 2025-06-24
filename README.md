# LLM fine-tuning with RL for Beam Design

## Vista Installation
```
git clone git@github.com:geoelements-dev/llm-rl.git
cd llm-rl
python -m ensurepip --upgrade
pip install virtualenv --user
virtualenv env
idev -m 30 -p gg -A BCS20003
source env/bin/activate
module load cuda/12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python3 beam-rl.py
```
