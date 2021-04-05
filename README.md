# CARRNN
A Continuous Autoregressive Recurrent Neural Network for Temporal Representation Learning
<br />

# Description
The continuous autoregressive recurrent neural network (CARRNN) can be applied to time series regression, prediction, and classification tasks to jointly model the trajectories of dynamic features or biomarkers, to predict the temporal developments, and to classify the clinical labels in sequential data with missing values. This toolbox is an implementation of the algorithm proposed in [1].
<br />

# Algorithm
The algorithm is developed for modeling multiple temporal features in sporadic data using an integrated deep learning architecture based on a recurrent neural network (RNN) unit and a continuous-time autoregressive (CAR) model. The CAR model is a generalized discrete-time autoregressive model that is trainable end-to-end using neural networks modulated by time lags to describe the changes caused by irregularity and asynchronicity.
<br />

# Dependencies
MATLAB (tested with v9.8), Statistics and Machine Learning Toolbox (tested with v11.7), Deep Learning Toolbox (tested with v8.5, if using Adam optimizer).
<br />

# Inputs
•	A CSV file containing timestamp/age information, labels, and measurements in columns under variable names 'SubjectID', 'Label', 'Age', and 'Features'. Missing labels and missing values need to be assigned as empty cell and NaN, respectively.
<br />
•	Proportion of validation and test subjects to all available subjects in data partitioning.
<br />
•	Network parameters including the number of layer nodes, type of RNN, and activation functions.
<br />
•	Optimization parameters including the optimizer, RNN time step, and regularization and update rules.
<br />
•	Evaluation metric for validation and test predictions.
<br />

# Outputs
•	Training performance displayed on a progressive plot.
<br />
•	Validation performance displayed on a progressive plot.
<br />
•	Testing performance printed to the command window.
<br />

# Citation
When you publish your research using this toolbox, please cite [1] as
<br />
<br />
@article{Ghazi2021,
<br />
  title = {{CARRNN}: {A} Continuous Autoregressive Recurrent Neural Network for Deep Representation Learning from Sporadic Temporal Data},
  <br />
  author = {Mehdipour Ghazi, Mostafa and S{\o}rensen, Lauge and Ourselin, S{\'e}bastien and Nielsen, Mads},
  <br />
  journal = {CoRR},
  <br />
  year = {2021},
  <br />
  volume = {abs/},}
<br />

# References
[1] Mehdipour Ghazi, M., Sørensen, L., Ourselin, S., and Nielsen, M., 2021. CARRNN: A Continuous Autoregressive Recurrent Neural Network for Deep Representation Learning
from Sporadic Temporal Data. arXiv preprint arXiv:.
<br />
[2] Mehdipour Ghazi, M., Nielsen, M., Pai, A., Cardoso, M.J., Modat, M., Ourselin, S., and Sørensen, L., 2019. Training recurrent neural networks robust to incomplete data: application to Alzheimer’s disease progression modeling. Medical Image Analysis 53, 39-46.
<br />

Contact: mostafa.mehdipour@gmail.com
<br />
