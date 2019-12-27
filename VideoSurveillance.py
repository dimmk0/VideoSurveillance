import configparser
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime



font = cv2.FONT_HERSHEY_SIMPLEX


class SaveVideo:
    def __init__(self, frame_width, frame_height, file='./output.avi'):
        self.output_file = file
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.out = cv2.VideoWriter(self.output_file, fourcc, 20.0, (frame_width, frame_height))
        print("Created new SaveVideo object")

    def save_frame(self, frame):
        self.out.write(frame)

    def __del__(self):
        print("Recording object is deleted")
        self.out.release()


class VideoProxy:
    def __init__(self):
        self.video_recording = None
        self.recording_created = False
        self.camshot_interval = 5
        self.camshot_start_time = time.time()

    def write_frame(self, frame):
        if self.recording_created:
            self.video_recording.save_frame(frame)
        else:
            frame_height, frame_width = np.array(frame)[:,:,0].shape
            recording_file_name = self.get_video_timestamp_filename()
            self.video_recording = SaveVideo(frame_width, frame_height, file=recording_file_name)
            self.video_recording.save_frame(frame)
            self.recording_created = True

    def write_shot(self, frame):
        if time.time() - self.camshot_start_time > self.camshot_interval:
            print("Creating shot")
            image_path = self.get_image_timestamp_filename()
            cv2.imwrite(image_path, frame)
            self.camshot_start_time = time.time()

    def get_video_timestamp_filename(self):
        return datetime.now().strftime("video\\record_%m_%d_%Y_%H.%M.%S.avi")

    def get_image_timestamp_filename(self):
        return datetime.now().strftime("shots\\camshot_%m_%d_%Y_%H.%M.%S.png")

    def stop_recording(self):
        if self.recording_created:
            del self.video_recording
            self.recording_created = False
            print("Recording is finished")


class Surveillance:
    def __init__(self, conf):
        config = configparser.ConfigParser()
        config.read(conf)
        self.configuration = config['video']

        self.video_source = eval(self.configuration.get('src'))
        self.record_duration = eval(self.configuration.get('record_duration'))
        self.sdThresh = eval(self.configuration.get('frame_change_threshold'))
        self.make_shots = eval(self.configuration.get('make_shots'))
        self.make_records = eval(self.configuration.get('save_video'))

        self.record_start_time = time.time() - self.record_duration
        self.recording = VideoProxy()

    def dist_map(self, frame1, frame2):
        """Calculate movement matrix then perform matrix normalization
        1)Разницей матриц (frame1 - frame2) мы вычисляем вектор движения
        который указыает куда переместился обьект с frame1 в frame2
        2)Рассчитываем расстояние между точками двух матриц(вычисляем длинну вектора)
        3)После чего нормализуем полученную матрицу
        """
        frame1_32 = np.float32(frame1)
        frame2_32 = np.float32(frame2)
        # 1
        diff32 = frame1_32 - frame2_32
        # 2 3
        norm32 = np.sqrt(diff32[:, :, 0] ** 2 + diff32[:, :, 1] ** 2 + diff32[:, :, 2] ** 2) / np.sqrt(
            255 ** 2 + 255 ** 2 + 255 ** 2)
        dist = np.uint8(norm32 * 255)
        return dist

    def run_detection(self):

        cap = cv2.VideoCapture(self.video_source)
        frame1 = cap.read()[1]
        frame2 = cap.read()[1]

        while True:
            frame3 = cap.read()[1]
            cv2.imshow('dist', frame3)
            mod = self.dist_map(frame1, frame3)
            frame1 = frame2
            frame2 = frame3

            # apply thresholding
            _, thresh = cv2.threshold(mod, 100, 255, 0)
            # calculate st dev test
            _, st_dev = cv2.meanStdDev(mod)

            cv2.imshow('dist', mod)
            cv2.putText(frame2, "Standard Deviation - {}".format(round(st_dev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

            if st_dev > self.sdThresh:
                self.record_start_time = time.time()

            if self.make_records:
                if time.time() - self.record_start_time < self.record_duration:
                    self.recording.write_frame(frame2)
                else:
                    self.recording.stop_recording()

            if self.make_shots and time.time() - self.record_start_time < self.record_duration:
                self.recording.write_shot(frame2)

            cv2.imshow('frame', frame2)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        conf_file = sys.argv[1]
    else:
        conf_file = os.path.splitext(__file__)[0]+'.conf'

    #src = 0
    video_recording = Surveillance(conf_file)
    video_recording.run_detection()



