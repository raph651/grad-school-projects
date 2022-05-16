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
* **Stock Price Prediction**<br>
  *Technologies*: Python, Pytorch, Numpy<br>
  *Topics*: deep learning, recurrent nueral networks (RNNs), gated recurrent unit (GRU), long short-term memory (LSTM)

---
### Musical Robot
<div class="verticalhorizontal">
    <img src="https://user-images.githubusercontent.com/91817346/168462109-775d4377-eb91-42da-ad46-148480049309.png" width ="300" height="290" alt="centered image" />
</div>

A project that helps identify the genre of an mp3 music file and discover music of similar genres. See the demo at: https://www.youtube.com/watch?v=6ErHy6OuTg4 or simply try it yourself on the website http://musicalrobot0.westus.azurecontainer.io/ !

In this project, I collaboted with three classmates to develop a web app named musical robot. The first task was to collect raw music data (mp3 files and their attributes). We downloaded a set of 8,000 song clips of 30 seconds from Free Music Archieve. These song files were well trimmed and each contained the highlight part of the song to keep the genre characteristics from it. Only 8 labels were considered for classification task, because there was a diversity of genre labels and most genre were overlapping.   

--- 
### Application of Fokas' Method 
<div class="verticalhorizontal">
    <img style="float: left;" src="Application of method of Fokas/3.PNG" width ="425" height="320" alt="centered image" />
    <img style="float: right;" src="Application of method of Fokas/2.PNG" width ="400" height="290" alt="centered image" />
</div>

This project was inspired by a course topic of Fokas' method that Prof. Bernard Deconinck introduced in AMATH 567: *Applied Complex Analysis* at University of Washington. The goal of the project was to implement an algorithm that solved for linear partial differential equations (PDEs) following the methodology introduced by Athanassios S. Fokas. 

In this project, I first went through the procedures of Fokas' Method, and developed an implementation script that solved an one-dimensional heat equation with advection and typical boundary conditions. I used another two methods-spectral and time-stepping for comparison. The code could be found in the Matlab file, and the report can be found in the notebook. I showed the solutions to the problem using different methods were almost identical, which agreed with my expectations. Then I showed the efficiency and generality of Fokas' Method. 

This was my final project for the course AMATH 581: *Scientific Computing* at University of Washington. From this course I learned how to numerically solve the initial value problems and boundary value problems for ODEs and PDEs using finite difference method, spectral method, and time-stepping method. Applications included fluid dynamics, stability analysis, and signal processing. 
