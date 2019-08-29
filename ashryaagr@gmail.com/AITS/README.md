# NASA Asteroid Classification

This repository is my solution for the third task by AITS.
Problem Statement :
Using the asteroids data <https://www.kaggle.com/shrutimehta/nasa-asteroids-classification>
perform a comparison (measured by the test accuracy and training time) between 
- using original data for training
- using principal components of the data for training.

## Dataset

The dataset is present in this repository in 2 forms
- principal components in nasa.csv
- Raw Data in Raw_Data/*.json 

### Content
The data is about Asteroids - NeoWs. NeoWs (Near Earth Object Web Service) is a RESTful web service for near earth Asteroid information. 
With NeoWs a user can: search for Asteroids based on their closest approach date to Earth, lookup a specific Asteroid with its NASA JPL small body id, as well as browse the overall data-set.

### Acknowledgements
Data-set: All the data is from the (http://neo.jpl.nasa.gov/). This API is maintained by SpaceRocks Team: David Greenfield, Arezu Sarvestani, Jason English and Peter Baunach.

### AIM
Find potential hazardous and non-hazardous asteroids
Find Features responsible for claiming an asteroid to be hazardous