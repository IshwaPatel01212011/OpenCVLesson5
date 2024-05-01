import cv2
import numpy as np

"""ghost = cv2.imread("Lesson 5/images/ghost.png")


grey_ghost = cv2.cvtColor(ghost, cv2.COLOR_BGR2GRAY)
common_blur_grey_ghost = cv2.GaussianBlur(grey_ghost, (3,3),0,0)
circle_detections = cv2.HoughCircles(common_blur_grey_ghost,cv2.HOUGH_GRADIENT, 1, 10,param1 = 50, param2 = 30, minRadius = 5, maxRadius = 70)
circles = np.uint16(np.around(circle_detections))
print(circles)

for circle in circles[0,:]:
    x = circle[0]
    y = circle[1]
    r = circle[2]
    cv2.circle(ghost, (x,y), r, (0,0,0), 3)
    cv2.imshow("Ghost with Circles", ghost)
    cv2.waitKey(0)"""

dots = cv2.imread("Lesson 5/images/dots.jpg")

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByConvexity = True
params.minConvexity = 0.4
params.filterByInertia = True
params.minInertiaRatio = 0.01

dector = cv2.SimpleBlobDetector_create(params)
print(dector)

key_points = dector.detect(dots)
blogs = cv2.drawKeypoints(dots, key_points, np.zeros((1,1)), (195,143,255), cv2. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Dots",blogs)
cv2.waitKey(0)

print(len(key_points))