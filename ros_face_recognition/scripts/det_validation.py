import rospy,time 
import rosbag
import cv2
from std_msgs.msg import Int32, String
from matplotlib import pyplot as plt 
from sensor_msgs.msg import Image
import ImageFile
import glob as glob
import face_recognition
from common import Face,Timeout
from process.FaceDetectionCv import FaceDetectionCv

cascadeHaarFaceDetection = FaceDetectionCv("/home/ruthz/catkin_ws/src/ros_face_recognition/config")

base_dir = '/home/ruthz/Desktop/2003'
folder_names = glob.glob(base_dir+'/*/*/') 
img_f = '/home/ruthz/Desktop/data_set/2002/08/26/big/img_265.jpg'
img = cv2.imread(img_f)
cnn_face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")
face_locations = face_recognition.face_locations(img)
haar_locations = cascadeHaarFaceDetection.processImg(img)

#print(face_locations)
#print(cnn_face_locations)
#print(haar_locations)
img1 =img
img2 = img
img3 = img
'''
for location in face_locations:
	img1 = cv2.rectangle(img, (location[3], location[0]), (location[1], location[2]), (255,0, 0), 2)##left, top, right, bottom 
	font = cv2.FONT_HERSHEY_DUPLEX
	img1 = cv2.putText(img, "HOG", (location[3] + 6, location[2] - 6), font, 0.5, (255, 255, 255), 1)

'''
for location in cnn_face_locations:
	img2 = cv2.rectangle(img, (location[3], location[0]), (location[1], location[2]), (255,0, 0), 2)##left, top, right, bottom 
	font = cv2.FONT_HERSHEY_DUPLEX
	img2 = cv2.putText(img, "CNN", (location[3] + 6, location[2] - 6), font, 0.5, (255, 255, 255), 1)

'''
for location in haar_locations:
	img2 = cv2.rectangle(img, (location[3], location[0]), (location[1], location[2]), (255,0, 0), 2)##left, top, right, bottom 
	font = cv2.FONT_HERSHEY_DUPLEX
	img2 = cv2.putText(img, "haar", (location[3] + 6, location[2] - 6), font, 0.5, (255, 255, 255), 1)
'''
def IOU(det,gtruth):
	xa = max(det[3],gtruth[0])
	ya = max(det[0],gtruth[3])
	xb = min(det[1],gtruth[2])
	yb = min(det[2],gtruth[1])
	
	print((det[3], det[0], det[1], det[2]),(gtruth[0],gtruth[3],gtruth[2],gtruth[1]))
	interarea = max(0, xb - xa + 1) * max(0, yb - ya + 1)
	det_area = (det[2] - det[0] + 1) * (det[1] - det[3] + 1)
	truth_area = (gtruth[1] - gtruth[3] + 1) * (gtruth[2] - gtruth[0] + 1)
	
	iou = interarea/ float(truth_area + det_area - interarea)
	#Sprint(interarea,truth_area,det_area,iou)	
	return iou  

def main():
	center_x = [105.249970,184.070915,340.894300]
	center_y = [87.209036,129.345601,117.498951]
	length = [67.363819,41.936870,70.993052]
	breadth = [44.511485,27.064477,43.355200]

	gt_truth = []
	t1 = [int(center_x[0]-breadth[0]),int(center_y[0]+length[0]),int(center_x[0]+breadth[0]),int(center_y[0]-length[0])]

	gt_truth.append(t1)

	t2 = [int(center_x[1]-breadth[1]),int(center_y[1]+length[1]),int(center_x[1]+breadth[1]),int(center_y[1]-length[1])]
	gt_truth.append(t2)

	t3 = [int(center_x[2]-breadth[2]),int(center_y[2]+length[2]),int(center_x[2]+breadth[2]),int(center_y[2]-length[2])]

	gt_truth.append(t3)

	#print(gt_truth)
	for gt in gt_truth:
		ious = []
		for location in cnn_face_locations:
			iou = IOU(location,gt)
			ious.append(iou)
		print(ious)
	for gt in gt_truth:
		img1 = cv2.rectangle(img, (gt[2], gt[3]), (gt[0], gt[1]), (0,255, 0), 1)

	plt.imshow(img1)
	plt.imshow(img3)
	plt.show()

if __name__ == '__main__':
    main() 



	
'''
for folder in folder_names:
		    image_files = glob.glob(folder+'big/*.jpg')
		    for imgs in [cv2.imread(img) for img in image_files]:
		    	face_locations = face_recognition.face_locations(imgs, number_of_times_to_upsample=0, model="cnn")

'''
