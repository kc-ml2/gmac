from setuptools import setup

# TODO: add openai/baselines
setup(
    name="hide",
    install_requires=[
        'torch',
        'tqdm',
        'termcolor',
        'tensorboardX',
        'matplotlib',
        'pillow',
        'gym',
        'gym[atari]',
        'sklearn',
        'psutil',
        'ray',
        'pybullet',
        'seaborn',
        'mpi4py'
    ]
)
