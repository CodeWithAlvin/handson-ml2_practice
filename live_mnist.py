import cv2
import pickle

with open("mnist_model.pkl","rb") as f:
	model=pickle.load(f)

video=cv2.VideoCapture(0)


while True:
	_,f=video.read()
	res=model.predict_proba([cv2.cvtColor(cv2.resize(f,(28,28)),cv2.COLOR_BGR2GRAY).flatten()])
	if res.all() < 0.5:
		cv2.putText(f,"No Digit Found",(25, 20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
	else:
		cv2.putText(f,f"{res.tolist()[0].index(max(res.tolist()[0]))}",(25, 20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
	cv2.imshow("Digit Detection",f)
	cv2.waitKey(1)