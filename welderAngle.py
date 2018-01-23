# Standard imports
import cv2
import numpy as np;
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from sklearn.cluster import MiniBatchKMeans
sift = cv2.xfeatures2d.SIFT_create()
MIN_MATCH_COUNT = 5

trailAvg = -30
avgSize = 10


def detectLines(imageEdges):
    angleThreshold = .1
    expectedAngle = 1.4
    # This returns an array of r and theta values
    lines = cv2.HoughLines(imageEdges,1,np.pi/180, 80)
    # The below for loop runs till r and theta values 
    # are in the range of the 2d array
    welderAngle = []
    for lineVals in lines:
        r, theta = lineVals[0]
        welderAngle.append(theta - np.pi/2)
        # if ((np.abs(theta) < angleThreshold) |\
        #  ((np.abs(theta) < np.pi/2 + angleThreshold) & (np.abs(theta) > np.pi/2 - angleThreshold)) |\
        #  ((np.abs(theta) < 3*np.pi/2 + angleThreshold) & (np.abs(theta) > 8*np.pi/2 - angleThreshold))):
        #     print(theta)
        #     continue
        if (np.abs(theta) < expectedAngle - 2*angleThreshold) | (np.abs(theta) > expectedAngle + angleThreshold):
            continue
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
     
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
         
        # x0 stores the value rcos(theta)
        x0 = a*r

        # y0 stores the value rsin(theta)
        y0 = b*r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be 
        #drawn. In this case, it is red.
        cv2.line(imageEdges,(x1,y1), (x2,y2), (255,0,0), 1)

    # All the changes made in the input image are finally
    # written on a new image houghlines.jpg
    return imageEdges, welderAngle



def grayQuantize(image, numValues):
    # load the image and grab its width and height

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 1))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = numValues)
    # print(clt)
    return clt

def applyQuant(image, clt):
    (h, w) = image.shape[:2]
    image = image.reshape((image.shape[0] * image.shape[1], 1))

    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 1))

    return quant

def preProcess(image):
    res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(res.mean())
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    res = clahe.apply(res)
    # gray = sharpen(gray)
    return res

def videoAnalysis(previous, current, clt):
    optMinX = 0
    optMaxX = 700
    optMinY = 0
    optMaxY = 2000

    contourArea = []
    welderOffEvent = False
    frameShiftEvent = False
    welderStopEvent = False

    height,width, depth = np.shape(current)



    try:
        gray = preProcess(current)
        previous = preProcess(previous)
    except Exception as e:
        print("Error")
        return False

    grayCopy = gray.copy()

    kernel = np.ones((20,20),np.float32)/400
    gray = cv2.filter2D(gray,-1,kernel)

    next = gray

    ret,thresh = cv2.threshold(gray, 140,255,cv2.THRESH_BINARY)
    threshcopy = thresh

    edges = cv2.Canny(grayCopy, 5, 100)

    imTemp, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]

    beadLocation = None

    if contours:
        M = cv2.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        contourArea.append(int(M["m00"]))
        if contourArea[-1] < 5000:
            welderOffEvent = True

        lineLength = 40

        cv2.line(thresh,(cX,cY + lineLength),(cX,cY - lineLength),(255,0,0),1)
        cv2.line(thresh,(cX + lineLength, cY),(cX - lineLength, cY),(255,0,0),1)

        ellipse = cv2.fitEllipse(contours[0])
        beadLocation = np.asarray(ellipse[0])

        welderRectangleCorner = beadLocation + np.asarray((ellipse[1][1]*0, 0))
        welderRectangle = np.asarray((welderRectangleCorner, np.asarray((600, -200)) + welderRectangleCorner))
        welderRectangle = welderRectangle.astype(np.int64)
        welderRectangle = tuple(map(tuple, welderRectangle))

        rect_img = np.zeros((height,width), np.uint8)
        cv2.rectangle(rect_img, welderRectangle[0], welderRectangle[1], (255,0,0), -1)
        masked_data = cv2.bitwise_and(grayCopy, grayCopy, mask=rect_img)
        masked_data = applyQuant(masked_data, clt)

        edges = cv2.Canny(masked_data, 5, 100)

        cv2.ellipse(thresh,ellipse,(255,0,0),2)

        hull = cv2.convexHull(contours[0])

        cv2.drawContours(thresh, contours, -1, (255,0,0), -2)
        cv2.drawContours(thresh, hull, -1, (255,0,0), -2)
        # cv2.imwrite('welderMask%d.png' %frameNum ,masked_data)
        # masked_data = cv2.bitwise_not(masked_data)
        cv2.imshow("Welder Mask", masked_data)

    res = cv2.add(thresh, grayCopy)
    res = cv2.add(res, edges)
    imageEdges, welderAngles = detectLines(edges)

    res = cv2.add(res, imageEdges)

    cv2.imshow("Contour", res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    res = {"welderAngles" : welderAngles,\
           "beadLocation" : beadLocation}
    return res


def main():
    cap = cv2.VideoCapture('sample_weld_video.mp4')    

    frameNum = 0

    ret, frame1 = cap.read()
    previous = frame1.copy()
    frame1 = preProcess(frame1)
    clt = grayQuantize(frame1, 4)

    while(cap.isOpened()):
        ret, current = cap.read()
        results = videoAnalysis(previous, current, clt)
        print(results)
        previous = current

    cap.release()
    cv2.destroyAllWindows()

    plt.plot(contourArea)
    plt.ylabel('Contour Area')
    plt.show()

if __name__ == "__main__":
    main()