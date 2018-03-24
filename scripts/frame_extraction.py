import cv2
import os
from os import listdir
from os.path import isfile, join
import pano

place_title = "boston_south_station_gopro"
root_dir  	= os.path.join(os.getcwd(), "..", "data", place_title)

left_dir    = os.path.join(root_dir, "1")
forward_dir = os.path.join(root_dir, "2")
right_dir   = os.path.join(root_dir, "3")
back_dir    = os.path.join(root_dir, "4")
top_dir     = os.path.join(root_dir, "5")

output_dir  	= os.path.join(root_dir, "stitched")

left_files 	    = [join(left_dir, f) for f in listdir(left_dir) if isfile(join(left_dir, f))]
forward_files 	= [join(forward_dir, f) for f in listdir(forward_dir) if isfile(join(forward_dir, f))]
right_files 	= [join(right_dir, f) for f in listdir(right_dir) if isfile(join(right_dir, f))]
back_files 		= [join(back_dir, f) for f in listdir(back_dir) if isfile(join(back_dir, f))]
top_files 		= [join(top_dir, f) for f in listdir(top_dir) if isfile(join(top_dir, f))]

if not len(left_files) == len(forward_files) == len(right_files) == len(back_files) == len(top_files):
	print("file count not consistent")
# print(left_files, forward_files, right_files, back_files, top_files)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

os.chdir(output_dir)

all_files = [left_files, forward_files, right_files, back_files, top_files]
total_cnt = 0
output_files = [""] * len(left_files)

for a, direction in enumerate(all_files):
	for i in range(len(direction)):
		vidcap = cv2.VideoCapture(direction[i])
		success, image = vidcap.read()
		count = 0
		success = True
		while success:
			success, image = vidcap.read()
			# print(count)
			if (count % 6 == 0):
				filename = "frame%s_scene%s_%s.jpg" % (count, i, str(a + 1))
				cv2.imwrite(filename, image)     # save frame as JPEG file
				print("for camera %s, scene %s, saved no.%s" % (str(a + 1), i, int(count / 6)))
				output_files[i] += (os.path.join(output_dir, filename) + " ")

			if cv2.waitKey(10) == 27:                     # exit if Escape is hit
				break
			count += 1

for i in range(len(left_files)):
	s = pano.Stitch(output_files[i][:-1])
	s.leftshift()
	# s.showImage('left')
	s.rightshift()
	print ("done")
	cv2.imwrite("scene%s.jpg" % i, s.leftImage)
	print ("image written")
	cv2.destroyAllWindows()
