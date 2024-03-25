# bb_rhythm

This repository contains a python package for analysing social behaviour and rhythmicity in honey bee colonies.

## Goal
The goal of this package is to provide a comprehensive implementation of methods for studying and analyzing behavioral 
rhythmicity in bees. The repository is offering a toolkit for data acquisition, preprocessing, and analysis, focusing 
on the following key objectives:
1) Data Acquisition and Preprocessing: Implement methods for collecting trajectory data of marked bees using the 
[BeesBook](10.3389/frobt.2018.00035) system. 
2) Cosinor Model of Individuals’ Activity: Develop algorithms for fitting cosine curves to individual bee movement 
speeds to analyze rhythmic expression and detect 24-hour activity patterns.
3) Interaction Modeling: Implement techniques for detecting and analyzing interactions between bees based on trajectory 
data, including timing, duration, and impact on movement speed.
4) Statistical Analysis and Validation: Conduct statistical tests to validate the significance of observed behavior 
patterns and interactions compared to null models and simulated data.
5) Visualization and Interpretation: Provide visualization tools to interpret and present the results of the analysis.
6) Weather Api and Time: Provide functionality and an api to get data from the *Deutscher Wetterdienst* and for converting 
suntime and utc-time.

## Structure
The source code is provided in the bb_rhythm folder. It contains several python submodules for creating data structures, 
data models, but also for analysing the data. Further functionality and use of the scripts can be found in [Usage](#usage).
```
bb_rhythm\
    \interactions.py
    \network.py
    \plotting.py
    \rhythm.py
    \statistics.py
    \time.py
    \utils.py
    \weather_api.py
```

## Installation
The package can be installed using ``pip``.
```
pip3 install git+https://github.com/BioroboticsLab/bb_rhythm.git
```

## Usage
### interactions.py
The scripts are 

More usage examples can be found in the [speedtransfer](https://github.com/BioroboticsLab/speedtransfer.git) repository.

