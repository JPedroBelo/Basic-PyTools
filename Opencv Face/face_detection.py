import cv2
import numpy as np
import dlib
from time import perf_counter
import os, sys
from deepface import DeepFace
from collections import Counter


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class EmotionRecognition:
	def __init__(self):
		self.NO_FACE = "no face"
		self.CONF_NO_FACE = 0.5


	def emotion_recogntion(self,frame,backend='opencv'):
		backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
		#image_channels = image.convert('RGB')
		#frame = np.array(image_channels)
		emotion_value = [self.NO_FACE,self.CONF_NO_FACE]
		rects = []
		try:
			blockPrint()
			analyze = DeepFace.analyze(img_path = frame,actions = ['emotion'],enforce_detection= True,detector_backend = backend,prog_bar = False)
			enablePrint()
			#print(analyze['age'])
			#print(analyze['gender'])
			#print(analyze['dominant_race'])
			region = analyze['region']

			x1 = region['x']
			y1 = region['y']
			x2 = region['x']+region['w']
			y2 = region['y']+region['h']
			#image1 = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
			#plt.imshow(image1)
			#plt.show()
			rects.append((x1,y1,x2,y2))
			emotion = analyze['dominant_emotion']
			emotion_conf = analyze['emotion'][emotion]
			emotion_value = [emotion,emotion_conf]
		except:
			enablePrint()
			emotion_value = [self.NO_FACE,self.CONF_NO_FACE]
		return emotion_value,rects

class FaceDetection:
	def __init__(self):
		self.dirr = "images/"


	def emotion_recogntion(self,frame,backend='opencv'):
		backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
		#image_channels = image.convert('RGB')
		#frame = np.array(image_channels)
		emotion_value = [self.NO_FACE,self.CONF_NO_FACE]
		rects = []
		try:
			blockPrint()
			analyze = DeepFace.analyze(img_path = frame,actions = ['emotion'],enforce_detection= True,detector_backend = backend,prog_bar = False)
			enablePrint()
			#print(analyze['age'])
			#print(analyze['gender'])
			#print(analyze['dominant_race'])
			region = analyze['region']

			x1 = region['x']
			y1 = region['y']
			x2 = region['x']+region['w']
			y2 = region['y']+region['h']
			#image1 = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
			#plt.imshow(image1)
			#plt.show()
			rects.append((x1,y1,x2,y2))
			emotion = analyze['dominant_emotion']
			emotion_conf = analyze['emotion'][emotion]
			emotion_value = [emotion,emotion_conf]
		except:
			enablePrint()
			emotion_value = [self.NO_FACE,self.CONF_NO_FACE]
		return emotion_value,rects



	def DNN_detection(self,image):
		modelFile = "opencv_face_detector_uint8.pb"
		configFile = "opencv_face_detector.pbtxt"
		net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
		frame = image.copy()
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		blob = cv2.dnn.blobFromImage(frame, 1.3, (750, 750), [104, 117, 123], False, False)
		conf_threshold = 0.5
		net.setInput(blob)
		detections = net.forward()
		bboxes = []
		rects = []
		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > conf_threshold:
				x1 = int(detections[0, 0, i, 3] * frameWidth)
				y1 = int(detections[0, 0, i, 4] * frameHeight)
				x2 = int(detections[0, 0, i, 5] * frameWidth)
				y2 = int(detections[0, 0, i, 6] * frameHeight)

				rects.append((x1,y1,x2,y2))
		return rects


	def Hog_detection(self,image):
		frame = image.copy()
		hogFaceDetector = dlib.get_frontal_face_detector()
		faceRects = hogFaceDetector(frame, 0)
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		rects = []
		for faceRect in faceRects:
			x1 = faceRect.left()
			y1 = faceRect.top()
			x2 = faceRect.right()
			y2 = faceRect.bottom()
			rects.append((x1,y1,x2,y2))
		return rects


	def MMOD_detection(self,image):		
		frame = image.copy()
		dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
		faceRects = dnnFaceDetector(frame, 0)
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		rects = []
		for faceRect in faceRects:
			x1 = faceRect.rect.left()
			y1 = faceRect.rect.top()
			x2 = faceRect.rect.right()
			y2 = faceRect.rect.bottom()
			rects.append((x1,y1,x2,y2))
		return rects



	def show_image(self,frame,rects,title="Image"):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		show = False
		for r in rects:
			show = True
			x1, y1, x2, y2 = r
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
		# Display
		if(show):

			cv2.imshow(title, frame)
			# Stop if escape key is pressed
			cv2.waitKey(0)

			# cv2.destroyAllWindows() simply destroys all the windows we created.
			cv2.destroyAllWindows()

def save_image(frame,rects=[],title="Image",label='Face'):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]
	show = False
	for r in rects:
		x1, y1, x2, y2 = r
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
		cv2.putText(frame,label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

	cv2.imwrite(title, frame)





def choose_emotion_by_conf(emotions):
	conf_emotions = {}
	for em in emotions:
		emotion = em[0]
		conf = em[1]
		if emotion in conf_emotions:
			conf_emotions[emotion] = (conf_emotions[emotion]+conf)/2
		else:
			conf_emotions[emotion] = conf

	em_max = max(conf_emotions, key=conf_emotions.get)
	return em_max

	


def test_image():
	faceDetec = FaceDetection()

	fileName = "teste.png"
	image = cv2.imread(fileName)
	rects = faceDetec.DNN_detection(image)
	faceDetec.show_image(image,rects)


def most_common(lst):
    return max(set(lst), key=lst.count)

def remove_from_list(value,array):
	lst = array[:]
	while value in lst: lst.remove(value)
	return lst

def test_many_images():
	faceDetec = FaceDetection()
	emotionRec = EmotionRecognition()

	face_count_DNN = 0
	face_count_Hog = 0 
	face_count_MMOD = 0
	count_emotions = 0
	N_IMG = 10
	N_SEQ = 8
	start_time = perf_counter()
	for i in range(1,N_IMG+1):
		emotion_step = []
		for j in range(1,N_SEQ+1):

			fileName = "image_"+str(i)+"_"+str(j)+".png"
			image = cv2.imread("Image/"+fileName)
			emotion = [emotionRec.NO_FACE,emotionRec.CONF_NO_FACE]
			#faceDetec.DNN_detection(image)
			
			#rects = faceDetec.Hog_detection(image)
			#if(len(rects)>0):
			#	face_count_Hog += 1

			rects = faceDetec.DNN_detection(image)
			#['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
			#emotion, rects = emotionRec.emotion_recogntion(image,backend='mtcnn')
			
			crop_img = image
			
			if(len(rects)>0):				
				face_count_DNN += 1
				x1,y1,x2,y2 = rects[0]
				sizeX = x2 - x1
				sizeY = y2 - y1

				square_side = max(sizeX,sizeY)*1.5

				meanX = int((x1+x2)/2)
				meanY = int((y1+y2)/2)
				newX1 = max(0,int(meanX-(square_side/2)))
				newX2 = min(int(meanX+(square_side/2)),image.shape[1])
				newY1 = max(0,int(meanY-(square_side/2)))
				newY2 = min(int(meanY+(square_side/2)),image.shape[0]) 



				crop_img = image[newY1:newY2, newX1:newX2]
				if( not (newY2<image.shape[0] and newX2<image.shape[1])):
					crop_img = image
				backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
				emotion, rects = emotionRec.emotion_recogntion(crop_img,backend='mtcnn')
				
				if(emotion[0]!=emotionRec.NO_FACE):
					count_emotions += 1

			
			emotion_step.append(emotion)

			save_image(crop_img,rects,os.path.join('results',fileName),label=emotion[0])
			#rects = faceDetec.MMOD_detection(image)
			#if(len(rects)>0):
			#	face_count_MMOD += 1
			end_time = perf_counter()
			#print(f'[{i}/{N_IMG}:{j}/{N_SEQ}] {end_time- start_time: 0.2f}')
			#faceDetec.show_image(image,rects)

		emotion = choose_emotion_by_conf(emotion_step)
		print("Step "+str(i)+" "+emotion)
		

	print("Faces Founded:")
	print("DNN: "+str(face_count_DNN))
	#print("HOG: "+str(face_count_Hog))
	#print("MMOD: "+str(face_count_MMOD))
	print("Emotions Founded: "+str(count_emotions))
	print(f'[{i}/{N_IMG}:{j}/{N_SEQ}] {end_time- start_time: 0.2f}')




if __name__ == "__main__":
	
	#test_image()
	test_many_images()