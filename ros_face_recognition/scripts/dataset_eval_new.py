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
import numpy as np
from time import time

from sklearn.metrics import classification_report, confusion_matrix

start = time()
cascadeHaarFaceDetection = FaceDetectionCv("/home/ruthz/catkin_ws/src/ros_face_recognition/config")
'''
base_dir = '/home/ruthz/Desktop/2003'
img_f = '/home/ruthz/Desktop/data_set/2002/08/13/big/img_1116.jpg'
img = cv2.imread(img_f)
cnn_face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")
face_locations = face_recognition.face_locations(img)
haar_locations = cascadeHaarFaceDetection.processImg(img)
'''



base_dir = '/home/ruthz/Desktop/data_set/'
files = '/home/ruthz/Desktop/data_set/FDDB-folds/file_names'
truths = '/home/ruthz/Desktop/data_set/FDDB-folds/ground_truth'

ground_truth_names = glob.glob(truths+'/*.txt') 
name_file_names = glob.glob(files+'/*.txt') 
name_file_names.sort()
ground_truth_names.sort()


def IOU(det,gtruth):
	xa = max(det[3],gtruth[0])
	ya = max(det[0],gtruth[3])
	xb = min(det[1],gtruth[2])
	yb = min(det[2],gtruth[1])
	
	#print((det[3], det[0], det[1], det[2]),(gtruth[0],gtruth[3],gtruth[2],gtruth[1]))
	interarea = max(0, xb - xa + 1) * max(0, yb - ya + 1)
	det_area = (det[2] - det[0] + 1) * (det[1] - det[3] + 1)
	truth_area = (gtruth[1] - gtruth[3] + 1) * (gtruth[2] - gtruth[0] + 1)
	
	iou = interarea/ float(truth_area + det_area - interarea)
	#print(interarea,truth_area,det_area,iou)	
	return iou


##get a dictionary of ground truth for each fold truth file
count = 0##if image file is missing count that
tp = 0;fp = 0;tn = 0;fn = 0;m_iou = 0;tot_iou = 0;tot_truth=0
y_pred = []; y_true = []
avg_iou = 0
estimated_iou = []
image_count = 0


##for each folds of the dataset 
##first storing a dictionary of groundtruth parsed from the text file
##and for each detected face iou is calculated 
for truth_file_name, file_name in zip(ground_truth_names,name_file_names):
	print(truth_file_name)
	file1 = open(truth_file_name,'r')
	lines = []
	gt_dict = {}#dictionary to store ground truth boxes for each file name
	for line in file1:
		#print("working") 
		lines.append(line.strip())
	i = 0	
	while(i<=len(lines)-1):	##from the ground truth file, get the bounding boxes and store it in a dictionary with the image file name as key	
		#print(i,lines[i])
		truth_boxes = []
		current_lines = lines[i+2:int(lines[i+1])+i+2]
		for bb_line in current_lines:
			#tot_truth = tot_truth+1
			bb = bb_line.split(" ")	
			##[x,y,length,breadth]
			truth_boxes.append([float(bb[3]),float(bb[4]),float(bb[0]),float(bb[1])])
		gt_dict[lines[i]] = truth_boxes
		i = i+int(lines[i+1])+2
	file2 = open(file_name,'r')
	bbs = []
	for file_name_line in file2:##use each image_file name as key to ground truth dictionary and evaluate IOU
		#print(gt_dict[file_name_line.strip()])
		img_name = base_dir+file_name_line.strip()+'.jpg'
		img = cv2.imread(img_name)
		image_count = image_count + 1
		if not np.shape(img):
			count = count+1			
			print(count,"no image",img_name)
			
		
		face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=2, model="cnn")
		#face_locations = face_recognition.face_locations(img,number_of_times_to_upsample=2)
		#face_locations = cascadeHaarFaceDetection.processImg(img)
		current_tp = 0	
		tot_truth = tot_truth + len(gt_dict[file_name_line.strip()])	
		gt_indices = []
		if(face_locations):
			for location in face_locations:
				ious = []## for one detection calculate ious with all ground_truth and take the maximum iou as a match. which is later thresholded.
				for truth in gt_dict[file_name_line.strip()]:
					iou = IOU(location,[int(truth[0]-truth[3]),int(truth[1]+truth[2]),int(truth[0]+truth[3]),int(truth[1]-truth[2])])
					ious.append(iou)
				if(np.amax(ious)>=0.3):##detection overlaps with gt
					if(gt_indices):
						if ious.index(np.amax(ious)) not in gt_indices:
							tp = tp+1##true positive
							y_pred.append(1)
							y_true.append(1)
							current_tp = current_tp +1
							estimated_iou.append(np.amax(ious))##use it for iou calculation
							gt_indices.append(ious.index(np.amax(ious)))
						else:
							fp = fp+1##false positive
							y_pred.append(1)
							y_true.append(0)
					else:
						tp = tp+1##true positive
						y_pred.append(1)
						y_true.append(1)
						current_tp = current_tp +1
						estimated_iou.append(np.amax(ious))##use it for iou calculation
						gt_indices.append(ious.index(np.amax(ious)))

				else:
					fp = fp+1##false positive
					y_pred.append(1)
					y_true.append(0)
			
			if(len(gt_dict[file_name_line.strip()])-current_tp!=0):	
				fn = fn + len(gt_dict[file_name_line.strip()])-current_tp##false negatives are total truthboxes in an image- true_positives in an image
				counter = 0 
				for ids in range(len(gt_dict[file_name_line.strip()])-current_tp):
					y_pred.append(0)#false negative count
					y_true.append(1)
					counter = counter+1
				if(counter!=len(gt_dict[file_name_line.strip()])-current_tp):
					print("weird",file_name_line.strip(), counter,len(gt_dict[file_name_line.strip()])-current_tp)
		

				
		if(current_tp==0):##no face is detected in the image
			fn = fn + len(gt_dict[file_name_line.strip()])
			for ids1 in range(len(gt_dict[file_name_line.strip()])):
				y_pred.append(0)#false negative count
				y_true.append(1)




end = time()
print('sum_iou = ', sum(estimated_iou), 'total = ', tp, 'miou = ',sum(estimated_iou)/tp)
print('true_positive = ', tp, 'false_positive = ', fp,'false_negative = ', fn,'total_true_face= ', tot_truth, 'total images= ', image_count)	
print('y_pred = ',len(y_pred),'y_true =',len(y_true))		

print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true,y_pred))


print("total time = ",end - start)































