# Coffee Maturity Classification using Convolutional Neural Networks and Transfer Learning


This work shows the combined use of multispectral image acquisition systems that generate
large amounts of data, together with convolutional neural networks that allow the extraction of information
from these images for identification and classification processes. In this case, we present the use of 5
different neural networks reported in the literature to present a benchmark in the classification of cherry
coffee fruits according to their ripening stage. A dataset released with this work for future research,
acquired with a custom-developed multispectral image acquisition system, was used. The comparison of
the different networks through different balances allows presenting an accuracy higher than 98% when
classifying about 600 coffee fruits in 5 different stages of maturation, and this with the objective of
providing the farmer with a very high-quality classification model of coffee fruits, providing security and
a viable method for the classification of coffee, in a more optimized and more accurate way.

## Folders
- **training with TL** this folder contains the codes required to perform the training with transfer learning and multispectral images (15 channels), which contains the python file (code_15channels_with_weights.py) in which the experiments will be run and the file (models_classification2.py) that will contain the models, to be called by the main code.

- **training without TL** this folder contains the codes required to perform the training without transfer learning with multispectral images (15 channels), which contains the python file (code_15channels_without_weights.py ) in which the experiments will be run and the file (models_classification5.py) that will contain the models, to be called by the main code.

## Requirements
This repository requires the following libraries and frameworks:

- TensorFlow 2.10.0
- scikit-learn
- numPy 
- OpenCV 
- Matplotlib
- Time
- os
- scikit-image
- glob


This repository was developed in the Python3 (3.9.12) programming language.


## Authors
Universidad Autonoma de Manizales (https://www.autonoma.edu.co/)

- Mario Alejandro Bravo-Ortiz 
- Esteban Mercado-Ruiz 
- Juan Pablo Villa-Pulgarin 
- Harold Brayan Arteaga-Arteaga
- Oscar Cardona
- Gustavo Isaza
- Raúl Ramos-Pollán
- Reinel Tabares-Soto 



## References

[1] 







