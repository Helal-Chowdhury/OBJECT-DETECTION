# OBJECT-DETECTION


 <img align="left" width="1000" height="500" src="https://github.com/Helal-Chowdhury/OBJECT-DETECTION/blob/main/finaljpg.jpg">
 
 ### Introduction
In this project, SSD based object detection framework is used. Object labels is given in Object.txt file to detect the object.


### How to create project environment and install packages:

Create Environment and Installation Packages

```bash
conda create --name <environment name> python=3.8
conda activate <environment name>
pip install -r requirements.txt
```
In case you have difficulties with installation of specific version of tensorflow and other package use the following commands to install:
```bash
pip install tensorflow==x.x.x --no-cache-dir
pip install <package name>==x.x.x  --no-cache-dir
```
## RUN the App
To run the app, Go to __FRONEND__ folder and shoot this command:              
```bash
streamlit run Object_Detection.py
```
## From Web UI 

 - upload image
 
 - click the **Detect Object** button to detect the object


