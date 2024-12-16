# SaliencyMapsAreEncoder
This is the experiment code for the paper "Saliency Maps Are Encoders: Analysing Logical Relations Towards Interpretation". It contains the logical dataset-framework ANDOR and uses multiple Saliency Methods to verify different metrics.

## ANDOR

The logical ANDOR dataset-framework can be found under modules/dataset_selecter.py

## Most important files for the experiments:

- saliencyCollect: Contains the main experiments.
- mixModel.yaml: Contains 2 of 3 dataset parameters for SEML experiments.
- mixModelSmall.yaml: Contains 1 of 3 dataset parameters for SEML experiments.
- resultResultProcessing.ipynb: Contains the plotting and evaluation of the experiment results

## Dependencies and installation guide

A list of all needed dependencies (other versions can work but are not guaranteed to do so):
- python==3.10.12
- conda==22.9.0
- pip==23.3

We suggest a fresh conda environment! <br>
Create a new environment and run the following lines:<br>

Adapt CUDA versions if needed: <br>
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia <br>
or: <br>
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 <br>

Fruther install: <br>

pip install sacred==0.8.4 seml==0.3.7 imbalanced-learn==0.11.0 dill==0.3.7 pyts==0.11.0 transformer_encoder==0.0.3 grad-cam==1.4.8 captum==0.6.0 numpy==1.24.4 scipy==1.11.3 scikit-learn==1.3.1 matplotlib==3.8.0 shap==0.45.1 einops==0.8.0 <br>


Note, sometimes a cv2 package is missing. In that case use: pip install opencv-python-headless <br>

To use the jupyter notebook: <br>
pip install --upgrade ipykernel jupyter notebook pyzmq

### How to run (over SEML)

The experiments are set up to work with SEML on our cluster. Change the .yaml files for parameter tuning. <br>

1. Set up seml with seml configure <font size="6">(yes you need a mongoDB server for this and yes the results will be saved a in separate file, however seml does a really well job in managing the parameter combinations in combination with slurm) </font>
2. Configure the yaml file you want to run. Probably you only need to change the number of maximal parallel experiments ('experiments_per_job' and 'max_simultaneous_jobs') and the memory and cpu use ('mem' and 'cpus-per-task').
3. Add and start the seml experiment. For example like this:
	1. seml saliencyCollect add mixModel.yaml
	2. seml saliencyCollect add mixModelSmall.yaml
	3. seml saliencyCollect start
4. Check with "seml saliencyCollect status" till all your experiments are finished 
5. Please find the results in the presults or filteredResults folder. It includes a dict which can be explored with the code in resultResultProcessing.ipynb

## Reference

Transformer and LRP implementations are taken and adapted from https://github.com/hila-chefer/Transformer-Explainability

## Cite and publications

This code represents the used model for the following publication:<br>
"Saliency Maps Are Encoders: Analysing Logical Relations Towards Interpretation" (TODO Link)

If you use, build upon this work or if it helped in any other way, please cite the linked publication.
