# import the necessary packages
from imutils import face_utils
import dlib
import cv2
from PIL import Image
import io, base64
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
 
# load the input image and convert it to grayscale
#data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

data = "data:image/jpeg;base64,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"

im = Image.open(io.BytesIO(base64.b64decode(data.split(',')[1])))
im.save("hasby.png")

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
		cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

	file = open("testfile.txt","a") 
	file.write('{"training": %s, "target": [1]}' % (temp_x)) 
	file.close()
 	
# show the output image with the face detections + facial landmarks
cv2.imwrite('hasil.png', image)
