# Raphael Liu grad-school-projects


My graduate study includes topics in Machine Learning, Deep learning, and Scientific Computing. Here are some of the projects I've done so far, with details below.
  
* **Musical Robot, music genere prediction app** <br>
  *Technologies*: Python, Pytorch, Tensorflow, Streamlit, Docker, Azure, Nginx<br>
  *Topics*: machine learning, classification, support vector machine (SVM), time-series, principal component analysis (PCA)
* **Multi-task Learning (MTL) in Deep Learning (In progress)**<br>
  *Technologies*: Python, Pytorch, Numpy, <br>
  *Topics*: deep learning, MTL, convolutional neural networks (CNNs), object detection, classification
* **Application of Method of Fokas**<br>
  *Technologies*: Matlab<br>
  *Topics*: advanced PDEs, method of Fokas, complex analysis 
* **Human-detection in video game *CS:GO* (In progress)**<br>
  *Technologies*: Python, YOLOv5, Pytorch, Numpy, OpenCV<br>
  *Topics*: deep learning, computer vision, time-series, classification 
* **DCGAN on training machine to write digits**<br>
  *Technologies*: Python, Pytorch, Numpy<br>
  *Topics*: deep learning, generative adversarial network (GAN), deep convolutional GAN (DCGAN), least square loss
* **Neural Style Transfer**<br>
  *Technologies*: Python, Pytorch, Numpy<br>
  *Topics*: deep learning, generative neural network (GNN), object-oriented programming, gramm matrix
* **Stock Price Prediction**<br>
  *Technologies*: Python, Pytorch, Numpy<br>
  *Topics*: deep learning, recurrent nueral networks (RNNs), gated recurrent unit (GRU), long short-term memory (LSTM)

---
### Musical Robot
<div class="verticalhorizontal">
    <img src="https://user-images.githubusercontent.com/91817346/168760849-5bde27df-2750-4ca2-923b-d3ece403f06a.png" width ="550" height="325" alt="centered image" />
</div>

A project that helps identify the genre of an mp3 music file and discover music of similar genres. See the demo at: https://www.youtube.com/watch?v=6ErHy6OuTg4 or simply try it yourself on the website http://musicalrobot0.westus.azurecontainer.io/ !

I collaborated with three classmates to develop a web app named musical robot for this project. The first task was to collect raw music data (mp3 files and their attributes). We downloaded a set of 8,000 song clips of 30 seconds from the Free Music Archive. These song files were well-trimmed, and each contained the highlighted part of the song to keep the genre characteristics from it. Only 8 labels were considered for the classification task because there were various genre labels, and most genres were overlapping. All the song files were decomposed to spectrograms and mel-frequencies. These features were mapped and trained by a SVM model for supervised learning. 

I utilized Streamlit to develop a website app that interacted with the user. The algorithm flow was first asking the user to upload a music file. This file would be decomposed into the features we have mapped. Once these features were input into our SVM model, the most matching genre would be returned and shown to the user. The user would have the opportunity to learn and listen to similar songs in that genre. The user would end the interaction after no longer needing the service. 

Since this prototype would play similar songs based on the user input, a repository of song files were needed. However, it was not realistic or convenient for users if they had to download all the song files (about 8 GB) to use this service. So I worked on deploying this app to a cloud hosting site. I utilized Docker to containerize the app. After I built the image, I  created an instance on the Microsoft Azure platform. Another Nginx file was added to reroute to Streamlit app port and to handle HTTP requests. Then this instance was deployed to the Azure cloud so that users could access the app service directly via the internet.

---
### Multi-task Learning (MTL) in Deep Learning (In progress)
![image](https://user-images.githubusercontent.com/91817346/170816526-8f7d6d93-fa28-4972-95b7-cbd28f2c075f.png)

This project aims to explore the most recent topics in Multi-Task Learning (MTL), and implement some of the most state-of-the-art MTL models such as MTAN and PeaceGAN. Also I want to analyze the impact of different weighting methods setting such as Random Loss Weighting, Impartial Multi-Task Learning. 

This project begins with applying MTL model on finger counts dataset from Kaggle. The dataset includes images of left and right hands with differernt hand guestures: 0, 1, 2, 3, 4, and 5. There are 12 classes in total if we do single-task learning, the most usual way. However, note that this task could be split into two: 'A': identifying left or right hand, and 'B': couting fingers. From this perspective, I implement a MTL model that begins with several shared convolutional layers, and then separates the last layer into two different branches. Use these two branches (FC layers) to complete task A and B, separately. The weighting method for this MTL model is RLW. The overall performance of this model reaches a 100% accuracy in 4 epoches, which is much more efficient than single-task learning.

The next part of the project aims to extend the MTL model to accomplish object detection, semantic segmentation, and object classification, (and object localization) at the same time. The datesets to use are NYUv2 and COCO's human activity images datasets. The implementation and presentation of this part are in progress.

--- 
### Application of Fokas' Method 
<div class="verticalhorizontal">
    <img style="float: left;" src="Application of method of Fokas/3.PNG" width ="425" height="320" alt="centered image" />
    <img style="float: right;" src="Application of method of Fokas/2.PNG" width ="400" height="290" alt="centered image" />
</div>

This project was inspired by a course topic of Fokas' method that Prof. Bernard Deconinck introduced in AMATH 567: Applied Complex Analysis at the University of Washington. The project's goal was to implement an algorithm that solved linear partial differential equations (PDEs) following the methodology introduced by Athanassios S. Fokas.

In this project, I first went through the procedures of Fokas' Method and developed an implementation script that solved a one-dimensional heat equation with advection and typical boundary conditions. I used another two methods-spectral and time-stepping, for comparison. You can find the code in the Matlab file and the report in the notebook. I showed that the solutions to the problem using different methods were almost identical, which agreed with my expectations. Then I showed the efficiency and generality of Fokas' Method.

This was my final project for the course AMATH 581: Scientific Computing at the University of Washington. This course taught me how to numerically solve the initial and boundary value problems for ODEs and PDEs using finite difference, spectral, and time-stepping methods. Applications included fluid dynamics, stability analysis, and signal processing.

---
### Human-Detection in video game: *CSGO* (In progress)


This project aims to identify the two teams-Terrorists and Counter-Terrorists in the popular video game: *CSGO* and localize the characters on the computer screen. 

The first part is to collect screenshots and assign location labels for each one. The label assigning tool used is Roboflow.

The second part is to implement a NN model with YOLOv5 toolkit. More details on this later.

---
### DCGAN on training machine to writte digits 

![download](https://user-images.githubusercontent.com/91817346/170817856-6c5df1ad-fcdb-478f-899b-10c5bf19b188.png)

---
### Neural Style Transfer 

---
### Stock Price Prediciton
