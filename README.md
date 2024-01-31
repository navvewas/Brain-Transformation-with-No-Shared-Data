# Brain-Transformation-with-No-Shared-Data


## Software Requirements
### OS Requirements
* Linux 7.6.1810

### Python Dependencies
* python 3.6
* tensorflow 1.14
* keras 2.2.4
* pytorch 1.1.0
* numpy 1.16.2
* pandas 0.24.2
* sklearn 0.20.3
* seaborn 0.9.0

## Hardware Requirements
These experiments were conducted on Quadro RTX 8000 (49GB). The code can run on any GPU with at least 42GB and CUDA
compatibility >=7.0.
Additionally, enough RAM to support in-memory operations is required (estimated 50G).

## Installation of code and enviorment
Run the following commands in **bash** (other shells requires some modification to the following scripts):
```bash
# download the code
git clone https://github.com/navvewas/Brain-Transformation-with-No-Shared-Data.git

# install conda env - assumes conda is already installed
conda create -c pytorch -c defaults -c conda-forge -n Brain2Brain --file env.yml
```
Depending on your internet connection, this may take about 30 minutes.

## Downaloading the data for the demo
Run the following commands in **bash** (other shells requires some modification to the following scripts):
```bash
# Function to extract file name from Dropbox URL
cd data
wget https://www.dropbox.com/scl/fi/plprfgnc5vdiwcp6zww7u/Demo_data.zip?rlkey=sqm73wigta98a2cgfyiptmrss&dl=0
unzip Demo_data.zip?rlkey=sqm73wigta98a2cgfyiptmrss&dl=0

```


## Demo - Transforamtion between datasets
This Demo will train transforamtion between subject 1 from the NSD dataset to subject 4 from the GOD dataset.
 To run the demo, run the following commands in **bash** (other shells requires some modification to the following scripts):
```bash
# go to the project directory
cd <PROJECT_DIR>
# activate conda environment
conda activate Brain2Brain

# run training code
python train_transformation.py 0 4 0 1 1 0 0 0 700 6400 1200 0

```
Once done, the weigths of the trained transforamtion can be found at the 'data/Transformations' folder.

## Demo - Train GOD encoder using another subject from NSD as a teacher

This Demo will train encoder of subject 4 from the GOD dataset using subject 1 from the NSD dataset as a teacher. We will train the encoder only with 300 examples in subject 4 in GOD.
 To run the demo, run the following commands in **bash** (other shells requires some modification to the following scripts):
```bash
# go to the project directory
cd <PROJECT_DIR>
# activate conda environment
conda activate Brain2Brain

# run training code
python train_encoder.py 0 4 0 1 1 0 300 0
```
Once done, the weigths of the trained transforamtion can be found at the 'data/Transformations' folder.



##
### Acknowledgments
- This code borrows from [Perceptual Similarity Metric](https://github.com/richzhang/PerceptualSimilarity).
- The original datasets behind the released data derivatives are ([fMRI on ImageNet](https://openneuro.org/datasets/ds001246/versions/1.0.1), and [ILSVRC](https://image-net.org/challenges/LSVRC/index.php)).
##
