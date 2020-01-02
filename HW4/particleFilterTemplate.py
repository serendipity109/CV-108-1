import cv2
import numpy as np
import math
 
def gaussian1D(m, s):
  r = np.random.normal(m,s,1) #s: standard deviation
  return r

def gaussian2D(m, s):
  r1 = np.random.normal(m,s,1) #s: standard deviation
  r2 = np.random.normal(m,s,1) #s: standard deviation
  return np.array([r1[0],r2[0]])

########   draw and show particles  #######
### numParticles : number of particles
### particles: particles state vectors
### frame: the image
### color: the particles color you assign
def display(numParticles, particles, frame, color): 
  #### TODO ####

cap = cv2.VideoCapture('person.wmv')

imgH = 480
imgW = 640

targetColor = np.array([0,0,255]) #red
colorSigma = #TODO set the value ###set sigma of color likelihood function, try 50 first
posNoise = #TODO  set the value ### set position uncertainty after prediction, try 15 first
velNoise = #TODO  set the value ### set velocity uncertainty after prediction, try 5 first

numParticles = 1000  #you can chage it 

#particle state vectors and initialization
particles = np.array([np.random.random_integers(0,imgW-1, numParticles), np.random.random_integers(0,imgH-1, numParticles), 3 * np.random.randn(numParticles) + 3, 3 * np.random.randn(numParticles) ] )
#weights and initalization
weights = np.zeros(numParticles) + 1.0/numParticles
#prediction matrix (constant velocity model)
predMat = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
frameCount = 0
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  rawFrame = np.copy(frame)
  if ret == True:
    frameCount = frameCount + 1
    if frameCount < 30:  ### start the particle filter from the 30th frames, simplify the process
      continue

    ##### calculate likelihood of the particles by pixel color and update the particle's weight
    for i in range(numParticles):
        # TODO...

    ##### Resampleing: the particle with higher weight will be duplicated more times
    ##### One suggested steps: 
    #       1. normalization (weights): make sum of all weights of particles to 1
    #       2. create a temporary state vectors (particleTemp)
    #       3. a loop (k) (loop through all particle state vector in "particleTemp" one by one)
    #         3-1 randomize an "particleIndex" by following the distribution reprsented by "weights" (if the particle with a higher weight, its index will be returned with a higher chance) (check the function numpy.random.choice() )
    #         3-2 copy the "particleIndex"-th state vector in "particles" to the "k-th" state vector in "particleTemp"
    #       4. copy whole particleTemp back to particles
    #       This steps may be easier to implement. But it is well approximate to the resampling step we say in the lecture.
    #### TODO......

    ##reset weights
    weights = np.zeros(numParticles) + 1.0/numParticles

    display(numParticles, particles, frame, (255,0,0)) #draw and show particles' location

    ##### predict new position of a particle by its velocity and old position (plus noise) (constant velocity model)
    for i in range(numParticles):
      #TODO....

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()