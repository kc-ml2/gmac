# Distributional Perspective on Actor-critic

For detailed code on GMAC refer to the following directories

- For [algorithms and loss](agents/gmac/train.py)
- For [network](agents/gmac/network.py)
- For [base framework](agents/a2c)
- For [other distributional versions](agents/iqac)

### Running
Please run the following to install dependencies
```
python setup.py install
```
Then, install openai baselines from 
https://github.com/openai/baselines
[[commit that was used](https://github.com/openai/baselines/tree/ea25b9e8b234e6ee1bca43083f8f3cf974143998)]


---
Sample command for running GMAC on atari breakout
```
python main.py --mode=gmac --env=atari --env_id=breakout --tag=test_run
```
