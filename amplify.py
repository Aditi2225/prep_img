delta = lambda_c / 8.0 / (1.0 + alpha)
exaggeration_factor = 2.0
lambda_ = sqrt(w * w + h * h) / 3

def amplify(src, spacialType):
    if spacialType == 0:
        curAlpha = lambda_ / delta / 8 - 1
        curAlpha *= exaggeration_factor

        if curLevel == levels or curLevel == 0:
            return src * 0
        else:
            return src * cv2.min(alpha, curAlpha)
    else:
        return src * alpha
