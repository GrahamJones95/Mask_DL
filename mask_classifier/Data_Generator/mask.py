import os
import sys
import random
import argparse
import numpy as np
from enum import Enum
from PIL import Image, ImageFile

__version__ = '0.3.0'


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, 'default-mask.png')
BLACK_IMAGE_PATH = os.path.join(IMAGE_DIR, 'black-mask.png')
BLUE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')
RED_IMAGE_PATH = os.path.join(IMAGE_DIR, 'red-mask.png')


def cli():
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    parser.add_argument('pic_path', help='Picture path.')
    parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    pic_path = args.pic_path
    if not os.path.exists(args.pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)

    if args.black:
        mask_path = BLACK_IMAGE_PATH
    elif args.blue:
        mask_path = BLUE_IMAGE_PATH
    elif args.red:
        mask_path = RED_IMAGE_PATH
    else:
        mask_path = DEFAULT_IMAGE_PATH

    FaceMasker(pic_path, mask_path, args.show, args.model).mask()


def create_mask(image_path,save_path,type):
    pic_path = image_path
    #mask_path = "/media/preeth/Data/prajna_files/mask_creator/face_mask/images/blue-mask.png"
    mask_path = BLUE_IMAGE_PATH
    show = False
    model = "hog"
    FaceMasker(pic_path, save_path, mask_path, show, model,type).mask()



class FaceMasker:
    KEY_FACIAL_FEATURES = ('top_lip','nose_bridge','chin')

    class Mask_Type(Enum):
        MASK_TYPE_CORRECT = 1
        MASK_TYPE_INCORRECT = 2
        MASK_TYPE_NOMASK = 3


    def __init__(self, face_path, save_path, mask_path, show=False, model='hog',type=Mask_Type.MASK_TYPE_CORRECT):
        self.face_path = face_path
        self.save_path = save_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.mask_type = type
        
    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        #self._face_img.show()
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        #print(face_landmarks.keys())
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            print(face_landmark.keys())
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)

        if found_face:
            if self.show:
                #self._face_img.show()
                1==1

            # save
            self._save()
        else:
            print('Found no face.')

    def _mask_face(self, face_landmark: dict):

        if(self.mask_type  == self.Mask_Type.MASK_TYPE_INCORRECT):
            """ top_lip = face_landmark['top_lip']
            nose_bridge = face_landmark['nose_bridge']

            top_lip = top_lip[len(nose_bridge)-1]
            nose_bridge = nose_bridge[len(nose_bridge)* 1 // 4]
            nose_point = ((top_lip[0]+nose_bridge[0])//2,(top_lip[1]+nose_bridge[1])//2) """
            
            top_lip = face_landmark['top_lip']
            nose_bridge = face_landmark['nose_bridge']

            #print(top_lip)
            #print(nose_bridge)
            nose_points = [point[1] for point in nose_bridge]
            min_index = np.argmax(nose_points)

            lip_point = top_lip[(len(top_lip) - 1) // 2]
            min_point = nose_bridge[min_index]

            #print(lip_point)
            #print(min_point)

            nose_point = ((lip_point[0]+min_point[0])//2,(lip_point[1]+min_point[1])//2)

            nose_v = np.array(nose_point)

            chin = face_landmark['chin']
            chin_len = len(chin)
            chin_bottom_point = chin[chin_len // 2]
            chin_bottom_v = np.array(chin_bottom_point)
            chin_left_point = chin[chin_len // 8]
            chin_right_point = chin[chin_len * 7 // 8]

            # split mask and resize
            width = self._mask_img.width
            height = self._mask_img.height
            width_ratio = 1.2
            new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

            # left
            mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
            mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
            mask_left_width = int(mask_left_width * width_ratio)
            mask_left_img = mask_left_img.resize((mask_left_width, new_height))

            # right
            mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
            mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
            mask_right_width = int(mask_right_width * width_ratio)
            mask_right_img = mask_right_img.resize((mask_right_width, new_height))

            # merge mask
            size = (mask_left_img.width + mask_right_img.width, new_height)
            mask_img = Image.new('RGBA', size)
            mask_img.paste(mask_left_img, (0, 0), mask_left_img)
            mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

            # rotate mask
            angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
            rotated_mask_img = mask_img.rotate(angle, expand=True)

            # calculate mask location
            center_x = (nose_point[0] + chin_bottom_point[0]) // 2
            center_y = (nose_point[1] + chin_bottom_point[1]) // 2

            offset = mask_img.width // 2 - mask_left_img.width
            radian = angle * np.pi / 180
            box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
            box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

            # add mask
            self._face_img.paste(mask_img, (box_x, box_y), mask_img)
        elif(self.mask_type  == self.Mask_Type.MASK_TYPE_CORRECT):
            nose_bridge = face_landmark['nose_bridge']
            nose_point = nose_bridge[len(nose_bridge) * 1 // 4]

            nose_v = np.array(nose_point)

            chin = face_landmark['chin']
            chin_len = len(chin)
            chin_bottom_point = chin[chin_len // 2]
            chin_bottom_v = np.array(chin_bottom_point)
            chin_left_point = chin[chin_len // 8]
            chin_right_point = chin[chin_len * 7 // 8]

            # split mask and resize
            width = self._mask_img.width
            height = self._mask_img.height
            width_ratio = 1.2
            new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

            # left
            mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
            mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
            mask_left_width = int(mask_left_width * width_ratio)
            mask_left_img = mask_left_img.resize((mask_left_width, new_height))

            # right
            mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
            mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
            mask_right_width = int(mask_right_width * width_ratio)
            mask_right_img = mask_right_img.resize((mask_right_width, new_height))

            # merge mask
            size = (mask_left_img.width + mask_right_img.width, new_height)
            mask_img = Image.new('RGBA', size)
            mask_img.paste(mask_left_img, (0, 0), mask_left_img)
            mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

            # rotate mask
            angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
            rotated_mask_img = mask_img.rotate(angle, expand=True)

            # calculate mask location
            center_x = (nose_point[0] + chin_bottom_point[0]) // 2
            center_y = (nose_point[1] + chin_bottom_point[1]) // 2

            offset = mask_img.width // 2 - mask_left_img.width
            radian = angle * np.pi / 180
            box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
            box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

            # add mask
            self._face_img.paste(mask_img, (box_x, box_y), mask_img)
        else:
            1 == 1

    def _save(self):
        str_path = self.face_path.replace('/','.')
        split = str_path.split('.')
        
        location = self.save_path + "without_mask/"
        if(self.mask_type == self.Mask_Type.MASK_TYPE_INCORRECT):
            location = self.save_path + "with_mask_incorrect/"
        elif(self.mask_type  == self.Mask_Type.MASK_TYPE_CORRECT):
            location = self.save_path + "with_mask_correct/"
        new_face_path = location + split[-2] + "-with_mask" + "." + split[-1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    #cli()
    create_mask(image_path)
