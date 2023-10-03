import cv2

def temporalIIRFilter(src, level):
    temp1 = (1.0 - cuttofFreqHigh) * lowpass1[level] + cutofFreqhigh * src
    temp2 = (1.0 - cuttofFreqlow) * lowpass2[level] + cutofFreqlow * src
    lowpass1[level] = temp1
    lowpass2[level] = temp2
    return lowpass1[level] - lowpass2[level]

def temporalIdalFilter(src):
    channels = [src[:,:,0], src[:,:,1], src[:,:,2]]
    
    for i in range(3):
        current = src[i]
        width = cv2.getOptimalDftSize(current.shape[1])
        height = cv2.getOptimalDftSize(current.shape[0])
        tempimg = cv2.copyMakeBorder(current, 0, height - current.shape[0], 0, width - current.shape[1])
        tempimg = cv2.dft(tempimg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE, tempimg.shape[0])
        filter_ = tempimg
        filter_ = createIdealBandpassFilter(filter_, cuttoffFreqLow, CuttofFreqHigh, rate)
        tempimg = cv2.mulSpectrums(tempimg, filter_, cv2.DFT_ROWS)
        tempimg = cv2.idft(tempimg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE, TEMPIMG.SHAPE[0])
        channels[i] = cv2.normalize(channels[i], 0, 1, cv2.NORM_MINMAX)
    
    return channels

def createIdealBandpassFilter(filter_, cuttoffFreqLow, CuttofFreqHigh, rate):
    FH = 2 * fh * width / rate
    for i in range(height):
        for j in range(width):
            if j >= f1 and j <= fh:
                response = 1.0
            else:
                response = 0.0
            filter_[i, j] = response
    return filter_
