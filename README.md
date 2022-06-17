# Raphael Liu grad-school-projects


My graduate study includes topics in Machine Learning, Deep learning, and Scientific Computing. Here are some of the projects I've done so far, with details below.
  
* **Musical Robot, music genere prediction app** <br>
  *Technologies*: Python, Pytorch, Tensorflow, Streamlit, Docker, Azure, Nginx<br>
  *Topics*: machine learning, classification, support vector machine (SVM), time-series, principal component analysis (PCA)
* **Multi-task Learning (MTL) in Deep Learning**<br>
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
  *Topics*: deep learning, generative neural network (GNN), object-oriented programming, gram matrix
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
### Multi-task Learning (MTL) in Deep Learning
<div class="verticalhorizontal">
    <img src="https://user-images.githubusercontent.com/91817346/170816526-8f7d6d93-fa28-4972-95b7-cbd28f2c075f.png" width ="750" height="325" alt="centered image" />
</div>

This project aims to explore the most recent topics in Multi-Task Learning (MTL) and implement some of the most state-of-the-art MTL models, such as MTAN and PeaceGAN. Also, I want to analyze the impact of different weighting methods setting such as Random Loss Weighting and Impartial Multi-Task Learning.

This project begins by applying the MTL model to Fingers dataset from Kaggle. This dataset includes images of hand gestures: 0, 1, 2, 3, 4, and 5. The hands can be either left or right hands. So there are 12 classes in total if we do single-task learning, the most usual way. However, note that this task could be split into two: 'A': identifying left or right hand, and 'B': counting fingers. From this perspective, I implement an MTL model that begins with several shared convolutional layers and then separates the last layer into two different branches. Use these two branches (FC layers) separately to complete tasks A and B. The weighting method for this MTL model is RLW. The overall performance of this model reaches a 100% accuracy in 4 epochs, which is much more efficient than single-task learning.

The next part of the project aims to extend the MTL model to accomplish semantic segmentation, depth estimation, and surface normal prediction on NUYv2 dataset. These 3 tasks are optimized during only one inference time. The NYUv2 dataset contains about 800 training images of indoor scenes captured by RGB and depth cameras. There are 13 classes in semantic segmentation - table, wall, floor... Therefore, a pixel-wise categorical cross-entropy loss is applied to the semantic segmentation task. For the depth estimation task, L1 loss is applied. And the cross-similarity loss is applied to the surface normal prediction task.

It is noted that even though the three tasks are related intuitively, each task loss can differ in magnitude. Thus, the model can suffer from the unbalance: one task converges quickly while the others converge too slowly. The random loss weighting (RLW) scheme is adopted to address this issue because convergence for RLW is almost always guaranteed. RLW also has a higher probability of escaping local minima compared to fixed loss weights, resulting in better performance. We use the ResNet-50 as the backbone or shared encoder and DeepLabv3+ architecture as task-specific decoders for all three tasks. We adopt the Multi-Task Attention Network (MTAN) approach to implement an attention module that shares features between encoder and decoders. The network is run on a GPU provided by Google Colab. The below shows three output predictions for a test image.

<div class="verticalhorizontal">
    <img src="https://user-images.githubusercontent.com/91817346/174393407-a6f922a7-5682-49b7-8403-f57c6e6417b6.png" width ="550" height="325" alt="centered image" />
</div>

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

A project that adopts the deep convolutional adversarial network (DCGAN) trains the machine to write MNIST hand-written digits. The network consists of two core parts: a discriminator that distinguishes between real and machine-generated(fake) images of hand-written digits and a generator that produces fake images from random noise.

The generator comes with a few FC layers, followed by 2d convolutional transpose layers. On the other hand, the discriminator begins with a few convolutional layers, followed by FC layers. I use the Least-Squares GAN loss for both the generator and the discriminator.
In my implementation, the training workflow is depicted as the following: first pass real images to the discriminator -> the discriminator gives scores on how real or fake they are -> pass random noise to generator to produce fake images -> pass fake images to the discriminator -> the discriminator gives scores again -> compute the discriminator loss & optimizer updates -> the generator produces new fake images and pass them to discriminator -> compute the generator loss & optimizer updates -> new epoch...

The losses for the discriminator and the generator decrease significantly at the beginning, especially for the generator. Then the two losses start to oscillate and are close to each other. Also, it seems that the generator loss is more dynamically changing, while discriminator loss is more stable. When the generator loss increases, the overall quality of fake images might decrease. The fake images generated over iterations is shown below.

<div class="verticalhorizontal">
    <img src="https://user-images.githubusercontent.com/91817346/170817856-6c5df1ad-fcdb-478f-899b-10c5bf19b188.png" width ="670" height="475" alt="centered image" />
</div>

---
### Neural Style Transfer 
<div class="verticalhorizontal">
    <img style="float: left;" src="NST/images/tubingen.jpg" width ="300" height="240" alt="centered image" />
    <img style="float: right;" src="NST/images/styler/starry_night.jpg" width ="300" height="240" alt="centered image" />
</div>

Thinking about transferring your favorite styling image to your pictures? This project makes use of generative neural networks to alter a source image towards a combination of itself with an arbitrary styler image. Users can customize the output by setting weights for the source image, styler image, and total variation loss (decides the smoothness of the output). Users can also choose which backbone to use (currently, VGG-16 and EfficientNet-B3 are supported) and tune the output by specifying the weights of layers.

To run an experiment, users can type in command line:

```Python
python NST_train.py --sp [styler image path] --cp[content image path] --backbone [vgg16 or efficientnet_b3] 
```

To get help with advanced configuration, type
```Python
python NST_train.py -h
```

The Neural Style Transfer (NST) method optimizes the whole content image by taking each pixel in that image as parameters instead of the hidden layers. The backbone extracts feature layers from the content image and the styler image. I first form the Gram matrices from both feature layers to compute the style loss. Then calculate the pixel-wise Euclidean distance between the Gram matrices. Since we are altering the content image, we get a newly generated image with the same shape as the content image after each step. So the content loss is computed as the pixel-wise Euclidean distance between the features layers extracted from the newly generated image and the content image. On the other hand, the total variational loss is given by the horizontal and vertical variations in the newly generated image. The weights of these losses are crucial, so they should be carefully tuned to achieve better visual performance. An example of transferring Van Gogh's *Starry Night* (picture on the right) style to the content image (picture on the left) is shown below.

<div class="verticalhorizontal">
    <img src="https://user-images.githubusercontent.com/91817346/174400449-955f3fef-8b1d-446e-9115-9f109847d1f4.png" width ="850" height="500" alt="centered image" />
</div>

---
### Stock Price Prediciton


