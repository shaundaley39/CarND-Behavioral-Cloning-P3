# **Behavioural Cloning** 

[//]: # (Image References)
[image1]: ./images/1.jpg "Track image"
[image2]: ./images/2.jpg "Track image"
[image3]: ./images/3.jpg "Track image"

### The Problem

The objective of this project a first exploration of end-to-end deep learning for driving a car: in this case, using Keras to build a neural network to drive a car around a simulator, while training that neural network using only examples of human driving.

The Udacity self-driving car simulator source code is available [on GitHub](https://github.com/udacity/self-driving-car-sim), along with compiled binaries and usage guidance. The course, in turn, is static - without moving obstacles or obstructions. To succeed, the car must "only" stay on the track at all times and navigate around a circuit (with varying road markings, edges, terrains and a bridge).

The simulated vehicle is quite simple - only the steering angle and throttle are required to navigate the car through its course. To simplifythis further, we use a set a constant target speed and use a proportional integral controller to control the throttle to pursue that target speed (see the implementation in [drive.py](drive.py)). All that must now be done, is to control the steering angle to keep the car on the road at all times, using only a camera feed from the front of the car. It is to this problem of controlling steering angle that behavioural cloning is to be applied.

![Track image 1][image1]
![Track image 2][image2]
![Track image 3][image3]

### Repository Structure

We mentioned [drive.py](drive.py) in the problem description above - this script is responsible for driving the car in autonomous mode, passing steering and throttle control messages to the simulator over a socket. This repository contains a number of further files. The neural network used by drive.py to determine steering angles, is persisted in [model.h5](model.h5). The python script that was used to define and train that model is [model.py](model.py).

Training data is kept off of GitHub due to storage cost, however a video of our model in action, driving the car autonomously around the simulator track, is provided in [video.mp4](video.mp4). The helper script used to generate that simulation video is [video.py](video.py).

And of course there is this README too, which serves as a project write-up.

### Solution Design and Model Architecture

To ensure efficient learning we wanted some image preprocessing - first, cropping away the top and bottom of each image from the camera feed since this is pure noise. The sky or mountains on the horrizon are not relevant to steering; in the best case they add noise that makes our learning harder, and in the worst case they act as landmarks to encourage track memorization and over-fitting. Secondly, image normalization (steering measurements are already effectively normalized - mean zeroed and mostly ranging -1.0 to 1.0).

To construct an appropriate neural network, we began by copying an architecture famously successful in precisely the domain we're working in: the Nvidia network applied in [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). What we first implemented in Keras was precisely this, except for one small change (not deliberate - a random mutation/ transcription error). Look close and you might notice the discrepency\*.

The neural network is entirely sequential:
- three stacked convolutional layers, each with a 5 by 5 kernel, 2 by 2 stride and relu activation. These layers have increasing depths of 24, 36 and 48 respectively
- two stacked convolutional layers with a 3 by 3 kernel, 1 by 1 stride and relu activation. These layers both have depth 64.
- flattening to a single dimensional vector
- four fully connected layers, with decreasing vector lengths of 100, 50, 10 and 1

To prevent over-fitting, we split the data 80%/ 20% into training and validation sets. We trained the model over multiple epochs, tracking relative mean square error (loss) across both training and validation sets. For regularization, we introduced random dropout (dropout probability: 50%) before each of the fully connected layers in our network.

Composing an appropriate data set was central to solving this problem, and could be covered extensively as part of the solution design. Instead we give data its own section (the next heading).

*\* well done for paying close attention! Our model has one less fully connected layer than the Nvidia model - and that's 1164 missing neurons. As you'll see, we acheive pretty good results and there's no imperative to further adjust the solution architecture now.*

### Training

#### The Data Set

Creating training data I was confronted with a harsh reality: my gaming reflexes aren't great. The first requirement was for me to drive several laps of the test circuit (crashing off the road and having to reset the simulation many times), learning to control and drive the simulation car.

With my own skills finally (somewhat) up to scratch, I set about recording several laps of smooth lane-centre driving, both clockwise and anti-clockwise directions around the circuit. That wasn't entirely smooth - I crashed or veered off a few times, and had to edit out those events in order to compose a training set of "correct" behaviour (like good parents).

I then went around several times recording only those portions of the course that were distinctive - for example the bridge, areas without road line markings or curves where the road is edged by sand banks. Since driving on these unusual sections requires concepts that might not be learned or generalized from the rest of the track, it is important to "over sample" driving in these environments.

Finally, I had to put the car into some environments that we never want it to get into (close to leaving the road), and demonstrate recoveries. Of course, we don't want recordings of things going wrong - only the recovery portion. We needed many extra recordings of the car moving from the edge of the road back into the centre (but without the frames where it's moving away from centre). We needed many recordings of the car moving onto a curve trajectory slightly late (but not the frames where the car was going off that trajectory for too long). And of course, while recording these recoveries, it was important to over-sample the more unusual portions of the track. This required some ambitious maneuvering, frequently toggling "record" on and off, lots of patience and quite a few restarts.

All in all, I collected 28693 data packets (each including an image from the perspective of a video camera on the front of the car; each including a steering angle measurement from that instant). Although further data was available, we only used this front facing camera image and the steering angle. These were then augmented by copying every image, flipping it left-right and taking the negative of the steering angle. This augmentation had the particular advantage of removing any left-right sampling bias from our training data (e.g. because we mostly drove around the track clockwise), hopefully preventing the network from learning "it's usually best to steer a little to the left, regardless of what we see" or some similarly awful lesson. Of course, this augmentation also had the general advantage of doubling our data set to 57386 points.

#### Training on the Model

A data set comprising 57386 images is quite massive, and cannot fit in memory. We were forced to write a generator that would load only the images for the current training batch, and train with this generator using the "fit_generator" model method in Keras. We trained with a batch size of 32, which is rather small (indeed, inefficiently so). Faster convergence ought to be possible with a larger batch size, but we attained decent results with this.

To train the model, we used the Adam optimizer minimizing mean squared error. We trained over 5 epochs, tracking both validation and training performance.

Now for the test.

### Results

It works - and remarkably well. Indeed, the car now drives itself better than I can drive it - after tirelessly training it on only the best examples of my driving, it seems reasonably robust and competent in driving smoothly and remaining on the road:

[![An autonomous drive around the simulated track](https://img.youtube.com/vi/2bnwBuGI49k/0.jpg)](https://www.youtube.com/watch?v=2bnwBuGI49k)
