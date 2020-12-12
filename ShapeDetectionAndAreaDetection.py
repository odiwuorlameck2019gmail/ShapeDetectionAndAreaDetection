import cv2 
import numpy as np 

#SET the font type.

font=cv2.FONT_HERSHEY_COMPLEX
#Read image from source.
img=cv2.imread("shape_detection.png")

#Convert the image into grayscale.

grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Deffine the image threshold.
ret,threshold=cv2.threshold(grayimage,239,256,cv2.THRESH_BINARY)

#get the contours .
contours,hierachy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Loop through all of the contours.

for contour in contours:
    approax=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    cv2.drawContours(img,[approax],0,(0),5)
    x=approax.ravel()[0]
    y=approax.ravel()[1]
    mx,my,mw,mh=cv2.boundingRect(contour)
    print("mw",mw)
    print("mh",mh)
    #Find area.
    area=cv2.contourArea(contour)
    area=(area*49)/74257.0
    if len(approax)==3:
        cv2.putText(img,"Triangle:{0:.1f} cm^2".format(area),(x,y),font,1,(255,0,255),1)
    elif len(approax)==4 and mw==mh:
        cv2.putText(img,"Squire:{0:.1f} cm^2".format(area),(x,y),font,1,(255,0,255),1)
    elif len(approax)==4:
        cv2.putText(img,"Rectangle:{0:.0f} cm^2".format(area),(x,y),font,1,(255,0,255),1)
    
    elif len(approax)==5:
        cv2.putText(img,"Pentagon:{0:.1f} cm^2".format(area),(x,y),font,1,(255,0,255),1)
    elif 6<len(approax)<15:
        cv2.putText(img,"Ellipse:{0:.1f}cm^2".format(area),(x,y),font,1,(255,0,255),1)
    else:
        cv2.putText(img,"Circle:{0:.1f}cm^2".format(area),(x,y),font,1,(255,0,255),1)
    
cv2.imshow("Shapes",img)
cv2.imshow("Threshold",threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()




