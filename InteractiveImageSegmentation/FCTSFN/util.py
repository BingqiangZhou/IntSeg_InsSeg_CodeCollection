import numpy as np

def getClickMap(points,shape):
    #print points.shape[1]; 
    tmpDist=255*np.ones((shape[0],shape[1]))
    [mx,my]=np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))       
    for i in range(points.shape[1]):
        tmpX=mx-points[i,0]
        tmpY=my-points[i,1]
        tmpDist=np.minimum(tmpDist,np.sqrt(np.square(tmpX)+np.square(tmpY)))
    tmpRst=np.array(tmpDist)
    tmpRst[np.where(tmpRst>255)]=255
    return tmpRst