# ConvE-STF MeteoNet - Marcille et. al 2023

## Introduction
This repository contains the code for reproducing the results of Marcille et al. 2024. It proposes a framework for short-term probabilistic wind forecasting. 
## Install environment
The environment requirements are contained in the file cnn_env.yaml. To initialize the environment using anaconda, run :

```
conda env create -f cnn_env.yaml 
```

## Data
The data files are provided as .pickle files. They are pytorch tensors containing the input and output data. 
- X_arome.pickle contains NWP forecasts. It is a dictionary {'train'; 'val'; 'test'} of tensors of shape (sample, variable, latitude, longitude, lead time)
- X_gs.pickle contains ground stations measurements. It is a dictionary {'train'; 'val'; 'test'} of tensors of shape (sample, variable, lead time)
- X_closest.pickle contains the NWP forecast at the closest grid point from the target. It is a dictionary {'train'; 'val'; 'test'} of tensors of shape (sample, variable, lead time)
- X_reduced.pickle contains the reduced dataset input. It is a dictionary {'train'; 'val'; 'test'} of tensors of shape (sample, variable, lead time).
- Y.pickle contains the target data. It is a dictionary {'train'; 'val'; 'test'} of tensors of shape (sample, variable, lead time).
- scalers.pickle contains the mean and standard deviations used for scaling the data. It is a dictionary containing scalers for NWP input, GS input, target. 

## Trained models
Trained models on the train-val-test split are provided in the trained_models folder. 
- AROME.pickle contains the NWP prediction at the closest grid point, corrected using ordinary least squares on the training dataset. 
- gbm_samples.pickle contains the predicted samples from the GBM baseline trained on the training dataset. The GBM model is too heavy to be shared in this repository, though the GBM.py file contains the details of the ```python class GBM_quantile``` class used for generating the model. 
- ConvE_STF_reduced.pickle contains the ConvE-STF-reduced model trained on the reduced dataset. It is an object of the class CNN1D_baseline_input in models.models.
- ConvE_STF.pickle contains the ConvE-STF model trained on the full dataset with gaussian output. It is an object of the class MNet_forecast_model_CNN in models.models.
- ConvE_STF_NF.pickle contains the ConvE-STF-NF model trained on the full dataset with normlaizing flows transform block. It is an object of the class MNet_forecast_model_CNN in models.models.

## Sampling and scoring
Functions for sampling and scoring the forecasts are compiled in scoring.scores and scoring.sampling. 

## Exemple
Results are reproduced in the notebook conve_stf_exemple.ipynb. 

## Citing
To cite this repository: 
```
@software{conve_stf_meteonet,
  author       = {Robin Marcille and
                  Pierre Tandeo and 
                  Maxime Thiébaut and 
                  Pierre Pinson and 
                  Ronan Fablet},
  title        = {Convolutional encoding and normalizing flows: a deep learning approach for offshore wind speed probabilistic forecasting},
  month        = jan,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {},
  url          = {}
}
```
This work is attached to a MIT License of use
## References

`conve_stf_meteonet` is attached to the following publication 
> R. Marcille, P. Tandeo, M. Thiébaut, P. Pinson, R. Fablet - Convolutional encoding and normalizing flows: a deep learning approach for offshore wind speed probabilistic forecasting. AIES. 2024
> [[arXiv]]

It uses the `nflows` library, and proposes a novel conditioned multivariate gaussian.
> C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, nflows: normalizing flows in PyTorch, Zenodo, 2020, 10.5281/zenodo.4296287

It uses code from [the AnDA repository](https://github.com/ptandeo/AnDA.git), published with 
> Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M., & Fablet, R. (2017). The Analog Data Assimilation. Monthly Weather Review, 145(10), 4093-4107
> [AMetSoc](http://journals.ametsoc.org/doi/abs/10.1175/MWR-D-16-0441.1)
