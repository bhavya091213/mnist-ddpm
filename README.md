# mnist_flow

MNIST generative modelling: DDPM vs rectified flow with a small UNet.

Status: work in progress.

## Run tests

    .venv/bin/python -m pytest -q
    .venv/bin/python -m pytest -q --cov=src/mnist_flow

Tests run on CPU and never download MNIST.
