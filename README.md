```markdown
# Wirehead Installation and Environment Setup

## Installing Wirehead

To install Wirehead, use the following command:

```bash
pip install wirehead -U
```

For more details, visit the official repository: [Wirehead GitHub Repository](https://github.com/neuroneural/wirehead)

---

## Method 1: Installing and Setting Up Nobrainer with Wirehead

1. Create a new Conda environment with Python 3.10:

    ```bash
    conda create -n wirehead python=3.10
    ```

2. Activate the newly created environment:

    ```bash
    conda activate wirehead
    ```

3. Install Wirehead:

    ```bash
    pip install wirehead -U
    ```

4. Clone the Nobrainer repository:

    ```bash
    git clone https://github.com/neuronets/nobrainer.git 
    ```

5. Navigate into the `nobrainer` directory:

    ```bash
    cd nobrainer
    ```

6. Checkout the `synthseg` branch:

    ```bash
    git checkout synthseg
    ```

7. Install Nobrainer in editable mode:

    ```bash
    pip install -e .
    ```

8. Move back to your original directory and restructure the Nobrainer directory:

    ```bash
    cd ..
    mv nobrainer nobrainer1
    mv nobrainer1/nobrainer nobrainer
    rm -r nobrainer1
    ```

9. Create a `log` directory in your main folder for storing logs:

    ```bash
    mkdir log
    ```

---

## Method 2: Installing a Different Training-Based Environment

If you want to install a different training-based environment, follow these steps:

1. Create a new Conda environment with Python 3.9:

    ```bash
    conda create --name torch2 python=3.9
    ```

2. Activate the newly created environment:

    ```bash
    conda activate torch2
    ```

3. Install Catalyst:

    ```bash
    pip3 install -U catalyst
    ```

4. Install `nccl` from the Anaconda channel:

    ```bash
    conda install -c anaconda nccl
    ```

5. Install additional required libraries:

    ```bash
    pip install mongoslabs
    pip3 install pymongo
    pip3 install nibabel
    pip3 install pynvml
    pip3 install scipy
    ```

For more information, you can visit the example repository: [Catalyst Example](https://github.com/neuroneural/catalyst_example)
```