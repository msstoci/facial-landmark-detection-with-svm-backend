# import the necessary packages
from imutils import face_utils
import dlib
import cv2
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
 
# load the input image and convert it to grayscale
image = cv2.imread("hasby.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
rects = detector(gray, 0)
 
# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
 
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	i = 0
	temp_x = []
	temp_y = []
	training = []
	for (x, y) in shape:
		i += 1
		temp_x.append(x)
		temp_y.append(y)
		#cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

	file = open("testfile.txt","a") 
	file.write('{"training": %s, "target": [1]}' % (temp_x)) 
	file.close()
 	
# show the output image with the face detections + facial landmarks
cv2.imwrite('hasil.png', image)