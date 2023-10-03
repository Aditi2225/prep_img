import cv2
import sys
import traceback
class WorkerFd(BaseWorker):
    def __init__(self, devid, frameBuffer, resultsFd, model_path, number_of_ncs):
        super().__init__(devid, model_path, number_of_ncs)
        self.frameBuffer = frameBuffer
        self.resultsFd = resultsFd

    def image_preprocessing(self, color_image):
        prepimg = cv2.resize(color_image, (300, 300))
        prepimg = prepimg[np.newaxis, :, :, :]
        prepimg = prepimg.transpose((0, 3, 1, 2))
        return prepimg

    def predict_async(self):
        try:
            if self.frameBuffer.empty():
                return
            color_image = self.frameBuffer.get()
            prepimg = self.image_preprocessing(color_image)
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1

                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_requests = []
                    self.inferred_cnt = 0

                self.exec_net.requests[reqnum].wait(-1)
                out = self.exec_net.requests[reqnum].outputs["detection_out"].flatten()
                detection_list = []
                face_image_list = []

                for detection in out.reshape(-1, 7):
                    confidence = float(detection[2])

                    if confidence > 0.3:
                        detection[3] = int(detection[3] * color_image.shape[1])
                        detection[4] = int(detection[4] * color_image.shape[0])
                        detection[5] = int(detection[5] * color_image.shape[1])
                        detection[6] = int(detection[6] * color_image.shape[0])

                        if (detection[6] - detection[4]) > 0 and (detection[5] - detection[3] > 0):
                            detection_list.extend(detection)
                            face_image_list.extend([color_image[int(detection[4]):int(detection[6])]])

                if len(detection_list) > 0:
                    self.resultsFd.put([detection_list, face_image_list])

                self.inferred_request[reqnum] = 0

        except Exception as e:
            traceback.print_exc()

