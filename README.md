# shopping_mall_train

## ENVIRONMENT
### Requirements
- Python 3.9

### Installation
0. Install poetry

    If you don't have `poetry`, get by running following command. <br>
    ※ Do not use `pip` to install `poetry` .
    ```
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    ```

1. Activate poetry environment
    ```
    poetry shell
    ```

2. Install dependencies
    ```
    poetry install
    ```

3. Install Pytorch

    Install PyTorch that works in your environmen. Normally, adding by following command.
    ```
    poetry add torch torchvision
    ```
    Or, if your GPU is NVIDIA RTX A6000 or something like "sm_86" architecture and CUDA version is 11.7, you have to adding by following command.
    ```
    poe pytorch
    ```