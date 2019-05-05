from webget import download  as wget
import matplotlib.pyplot as plt
import cv2

img_url = "https://static.independent.co.uk/s3fs-public/thumbnails/image/2017/10/09/11/faces-1.jpg"

def process():
	img = cv2.imread(wget(img_url))

	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces =  face_cascade.detectMultiScale(grey)
	
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_grey = grey[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_grey)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	
	tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.imshow(tmp)
	
if __name__ == '__main__':
	
	process()
	plt.show()