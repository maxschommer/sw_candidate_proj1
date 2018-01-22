# Standard imports
import cv2
import numpy as np;
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

sift = cv2.xfeatures2d.SIFT_create()
MIN_MATCH_COUNT = 5

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def siftMatch(targetImage):
    # find the keypoints and descriptors with SIFT
    sampleImage = cv2.imread('welderMask1.png',0)

    kp1, des1 = sift.detectAndCompute(sampleImage,None)
    kp2, des2 = sift.detectAndCompute(targetImage,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = sampleImage.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        targetImage = cv2.polylines(targetImage,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(sampleImage,kp1,targetImage,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()


def main():
    cap = cv2.VideoCapture('sample_weld_video.mp4')

    contourArea = []

    frameNum = 0

    ret, frame1 = cap.read()
    height,width,depth = frame1.shape

    

    


    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):

        frameNum += 1
        ret, frame = cap.read()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        except Exception as e:
            break

        grayCopy = gray.copy()




        kernel = np.ones((20,20),np.float32)/400
        gray = cv2.filter2D(gray,-1,kernel)

        next = gray

        imgDiff = mse(prvs, next)
        if imgDiff > 300:
            print(imgDiff)

        imgDiff2 = ssim(prvs, next)
        if imgDiff2 < .95:
            print(imgDiff2)

        ret,thresh = cv2.threshold(gray, 140,255,cv2.THRESH_BINARY)
        threshcopy = thresh

        # edges = cv2.Canny(grayCopy, 5, 40)

        imTemp, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]

        if contours:
            M = cv2.moments(contours[0])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            contourArea.append(int(M["m00"]))
            lineLength = 40

            cv2.line(thresh,(cX,cY + lineLength),(cX,cY - lineLength),(255,0,0),1)
            cv2.line(thresh,(cX + lineLength, cY),(cX - lineLength, cY),(255,0,0),1)

            ellipse = cv2.fitEllipse(contours[0])

            welderRectangleCorner = np.asarray(ellipse[0])+np.asarray((ellipse[1][1]/2, 0))
            welderRectangle = np.asarray((welderRectangleCorner, np.asarray((600, -200))+welderRectangleCorner))
            welderRectangle = welderRectangle.astype(np.int64)
            welderRectangle = tuple(map(tuple, welderRectangle))

            print(welderRectangle)

            rect_img = np.zeros((height,width), np.uint8)
            cv2.rectangle(rect_img, welderRectangle[0], welderRectangle[1], (255,0,0), -1)
            masked_data = cv2.bitwise_and(grayCopy, grayCopy, mask=rect_img)
            siftMatch(masked_data)
            # laplacian = cv2.Laplacian(masked_data, cv2.CV_64F)
            # sobelx = cv2.Sobel(masked_data,cv2.CV_64F,0,1,ksize=5)

            cv2.ellipse(thresh,ellipse,(255,0,0),2)

            hull = cv2.convexHull(contours[0])

            cv2.drawContours(thresh, contours, -1, (255,255,0), 3)
            cv2.drawContours(thresh, hull, -1, (255,255,0), 3)
            cv2.imwrite('welderMask%d.png' %frameNum ,masked_data)
            # masked_data = cv2.bitwise_not(masked_data)
            cv2.imshow("Welder Mask", masked_data)

        res = cv2.add(thresh, grayCopy)
        # res = cv2.add(res, edges)


        cv2.imshow("Contour", res)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prvs = next

    cap.release()
    cv2.destroyAllWindows()

    plt.plot(contourArea)
    plt.ylabel('Contour Area')
    plt.show()

if __name__ == "__main__":
    main()