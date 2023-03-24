# MLGameControl
Welcome to this repo! Here I have used Google's Mediapipe Hands to create a machine learning module for a hand tracking game controller
that interprets the location of your hand in the camera into the controls: up, down, left, right. 
And it all happens on the CPU!

![Screen Shot 2023-02-01 at 7 36 11 PM](https://user-images.githubusercontent.com/91762926/216210618-5eef1586-1f7f-48d1-9f0e-c7057c7db771.png)


## What is MediaPipe Hands?
Traditionally to train your own hand tracking model you would need as many samples of hands as you can get (with a variety on backgrounds, skintones, lighting, ect) to push through your ML pipeline with time and GPU.




However, MediaPipe has done the heavy lifting here to allow this game controller to function in on-device and in real time. Essentially using this architecture:

![This is an image](https://1.bp.blogspot.com/-s_rfpl9S-sQ/XVrS_bzhcKI/AAAAAAAAEhc/_OrSe3VDLt8o1L6l2mA5HJsaqVZdaObpgCEwYBhgL/s1600/image5.png)

To place 20 3-D hand landmarks (very accuratly) (average precision of 95.7%)

![This is an image](https://1.bp.blogspot.com/-8SxmsK5VoJ0/XVrTpMrJDFI/AAAAAAAAEiM/nAa3vuj8a2sjgEPAeMKXD4m09yKUgjVIQCLcBGAs/s640/Screenshot%2B2019-08-19%2Bat%2B9.51.25%2BAM.png)

read wayyyy more detailed description here: 
https://www.arxiv-vanity.com/papers/2006.10214/

## Purpose
Each of the 20 3D Landmarks:

![This is an image](https://mediapipe.dev/images/mobile/hand_landmarks.png)


output their corrdinates in relation to landmark 0 at a speed of about 13/FPS once a hand is detected 

![Screen Shot 2023-02-01 at 7 35 54 PM](https://user-images.githubusercontent.com/91762926/216209567-3feb025b-009a-4632-9cc2-92df1bbed158.png)


My goal was to focus using the coordinates in reference to the 2-D plane they are on. Instead of the 20 landmarks, I only took landmark 0 (at the base of the hand). 
This module tracks the base of the hand in relation to the screen on a plane of about 1200x720 to develop the controls: UP, DOWN, LEFT, RIGHT.

watch me demo it here: https://youtu.be/upNaenRRoZs

hints: 
* keep maxNumHands = 1 to avoid confusion (however you may switch between hands while it is running)
* SCOOT BACK in your camera for more accuracy
* remember the base of your palm is what is being tracked (landmark 0)



