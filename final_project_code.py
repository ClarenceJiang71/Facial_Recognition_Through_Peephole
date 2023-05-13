import cv2
import numpy as np
import face_recognition
from scipy import ndimage
import torch
import torch.nn.functional as F
import mediapipe as mp
from FaceAnalyzer import FaceAnalyzer, Face
import pickle
import pandas as pd
# from mtcnn import MTCNN


image_faceLocation_dict = {}
image_faceEncoding_dict = {}


def load_image(image_address):
    """
    Load and return image

    :return: a numpy array
    """
    img = cv2.imread(image_address)
    return img


def show_image(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_image(image, image_path):
    cv2.imwrite(image_path, image)


def image_crop(image):
    height, width = image.shape[0], image.shape[1]
    sub_image = image[int(height*0.25):int(height * 0.75), int(width * 0.25) : int(width*0.75)]
    return sub_image


def data_reduction(image, ratio):
    """
    Reduced an original image into a smaller size (0.1*original width, 0.1* original height).

    :return: a resized image with dimension (0.1 * original height, 0.1* original width)
    """
    original_height, origin_width = image.shape[0], image.shape[1]
    new_height = int(original_height*ratio)
    new_width = int(origin_width*ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def check_color_distribution(resized_image):
    """
    Explore the color distribution to figure out thresholds that could be applied to convert the image into
    a binary form

    :return: a numpy array with values equal to average of RGB
    """
    converted_2d_average = np.empty([resized_image.shape[0], resized_image.shape[1]])
    for i in range(resized_image.shape[0]):
        for j in range(resized_image.shape[1]):
            value = np.mean(resized_image[i][j])
            converted_2d_average[i][j] = value
    return converted_2d_average


def convert_to_binary(image, threshold):
    """
    """
    image_copy = image.copy()
    # upper_left_door = 0
    for i in range(image_copy.shape[0]):
        for j in range(image_copy.shape[1]):
            # filter out the white background on the corridor
            if np.mean(image_copy[i][j]) > threshold:
                image_copy[i][j] = 255
            # elif np.mean(image_copy[i][j]) <= 60:
            #     upper_left_door = ()
            else:
                image_copy[i][j] = 0
    return image_copy


def face_detection_fr(image):
    # A function that tries to do face detection using the face_recognition library.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_location = face_recognition.face_locations(image_rgb)[0]
    cv2.rectangle(image, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (255, 0, 255), 10)
    show_image(image)
    print(face_location)

    #
    # face_landmarks_list = face_recognition.face_landmarks(image)
    #
    # # Print the locations of the facial landmarks
    # for face_landmarks in face_landmarks_list:
    #     for landmark_name, landmark_list in face_landmarks.items():
    #         print(landmark_name, landmark_list)
    #
    # # Draw the facial landmarks on the image
    # for face_landmarks in face_landmarks_list:
    #     for landmark_name, landmark_list in face_landmarks.items():
    #         for (x, y) in landmark_list:
    #             cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # # Display the image with facial landmarks
    # cv2.imshow("Facial Landmarks", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def face_detection_face_analyzer(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fa = FaceAnalyzer()
    fa.process(img)

    # Now you can find faces in fa.faces which is a list of instances of object Face
    if fa.nb_faces > 0:
        print(f"{fa.nb_faces} Faces found")
    for face in fa.faces:
        eye_distance = face.getEyesDist()
        print(eye_distance)
        face.draw_bounding_box(img)
        show_image(img)


def face_detection_mtcnn(image):
    pass


def face_detection_refinaface(image):
    # detector = MTCNN()
    # faces = detector.detect_faces(image)
    # print(len(faces))
    # for face in faces:
    #     x, y, w, h = face['box']
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # # Show image with bounding boxes
    # cv2.imshow("Detected Faces", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass


def face_detection_haarcascade(img):
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # cannot classify people looking down
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') # give inaccurate face location for individual far away
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml') # give in accuracte face location for indiviual far away
    # face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml') # give in accuracte face location for indiviual far away
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    else:
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the output
        show_image(img)
        return (x, y, x+w, y+h)


def image_enhance_cv(img):
    scale_factor = 2
    interpolation = cv2.INTER_CUBIC
    new_size = (img.shape[1] * scale_factor, img.shape[0] * scale_factor)
    resized_img = cv2.resize(img, new_size, interpolation=interpolation)
    cv2.imshow('Original', img)
    cv2.imshow('Resized', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_enhance_scipy(img):
    scale_factor = 2
    resized_img = ndimage.zoom(img, (scale_factor, scale_factor, 1), order=3)
    cv2.imshow('Original', img)
    cv2.imshow('Resized', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_enhance_pytorch(img):
    scale_factor = 2
    img_tensor = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    upsample = F.interpolate(img_tensor, scale_factor=scale_factor, mode='bilinear')
    resized_img = upsample.squeeze(0).permute(1, 2, 0).cpu().numpy().astype('uint8')
    cv2.imshow('Original', img)
    cv2.imshow('Resized', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_enhancing_process(img):
    image_enhance_cv(img)
    image_enhance_scipy(img)
    image_enhance_pytorch(img)


def detect_person(image):
    # This method helps detect the main body part of th
    pose_detection = mp.solutions.pose.Pose()
    results = pose_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Extract landmarks for first detected pose
    pose_landmarks = results.pose_landmarks.landmark

    # Compute bounding box
    xmin, ymin, xmax, ymax = float('inf'), float('inf'), float('-inf'), float('-inf')
    for landmark in pose_landmarks:
        x, y = landmark.x * image.shape[1], landmark.y * image.shape[0]
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x)
        ymax = max(ymax, y)



    # Draw bounding box on image
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    cv2.imshow('Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    coordinates = (int(xmin), int(ymin), int(xmax), int(ymax))
    return coordinates


def check_surronding(binary_image, i, j):
    plus = 1
    while plus < 25:
        if binary_image[i, j+plus] != 255:
            return False
        plus += 1
    return True

    # if i-1 >= 0 and i+1 <= binary_image.shape[0] and j-1>=0 and j+1 <= binary_image.shape[1]:
    #     a = binary_image[i-1, j] == 255
    #     b = binary_image[i+1, j] == 255
    #     c = binary_image[i, j+1] == 255
    #     d = binary_image[i, j-1] == 255
    #     e = binary_image[i-1, j-1] == 255
    #     f = binary_image[i+1, j-1] == 255
    #     g = binary_image[i-1, j+1] == 255
    #     h = binary_image[i+1, j+1] == 255
    #     return a or b or c or d or e or f or g or h
    # return False


def head_locate_from_body(image, coordinates):
    # middle_index = int((coordinates[0] + coordinates[2])/2)
    half_head_width = int((coordinates[2]-coordinates[1])/4.5)

    body_image = image[coordinates[1]:int(coordinates[3]), coordinates[0]:coordinates[2]]
    # show_image(image)
    avgRGB_image = check_color_distribution(body_image)
    binary_image = convert_to_binary(avgRGB_image, 170)
    show_image(binary_image)

    # Find neck through the binary image
    rightmost_first_white_index = 0
    neck_y_index = 0
    for i in range(int(binary_image.shape[0]/2)):
        for j in range(binary_image.shape[1]):
            if binary_image[i][j] == 255:
                if check_surronding(binary_image, i, j) and j > rightmost_first_white_index:
                    rightmost_first_white_index = j
                    neck_y_index = i
                    break
                elif check_surronding(binary_image, i, j) and j <= rightmost_first_white_index:
                    break

    temp = rightmost_first_white_index
    while binary_image[neck_y_index][temp] == 255:
        temp += 1



    # cv2.rectangle(image, (coordinates[0], coordinates[1]-neck_y_index), (coordinates[2], neck_y_index+coordinates[1]), (0, 0, 255), 2)
    # cv2.rectangle(image, (middle_index-half_head_width, coordinates[1]-neck_y_index), (middle_index+half_head_width, neck_y_index+coordinates[1]), (0, 0, 255), 2)
    cv2.rectangle(image, (coordinates[0] + temp-half_head_width*2, coordinates[1]-neck_y_index), (coordinates[0] + temp, neck_y_index+coordinates[1]), (0, 0, 255), 2)
    # cv2.rectangle(image, (middle_index-half_head_width, int(coordinates[1])), (middle_index+half_head_width, int(coordinates[3])), (0, 255, 0), 2)
    show_image(image)
    return (coordinates[0] + temp-half_head_width*2, coordinates[1]-neck_y_index, coordinates[0] + temp,
            neck_y_index+coordinates[1])


def image_preparation(image_address):
    """
        Step 1: preparation
        1. load image
        2. crop the image to focus more on the person inside image
        3. reduce the image into smaller size
        4. store the new reduced image
        """
    original_image = load_image(image_address)
    # # image_enhancing_process(original_image)
    #
    cropped_image = image_crop(original_image)
    reduced_image = data_reduction(cropped_image, 0.1)
    # show_image(reduced_image)
    end_index = image_address.index(".jpg")
    write_path_address = image_address[:end_index] + "_r" + ".jpg"
    write_image(reduced_image, write_path_address)


def face_detection_solve_where(image_address):
    """
       Step 2: solving where the face is
       1. First detect the body (it is hard to directly apply some other face detection methods)
           -insight: size is even more critical than resolution
       2.
       """
    # Individual test
    # image = load_image(f"../images/image6_e_r.jpg")
    # coordinates = detect_person(image)
    # head_locate_from_body(image, coordinates)

    # Multiple image test on non-frontal face situation
    # for i in range(5, 7):
    #     # if i == 3:
    #     #     continue
    #     image = load_image(f"../images/image{i}_e_r.jpg")
    #     coordinates = detect_person(image)
    #     head_locate_from_body(image, coordinates)


    # Process each image based on different sets of methods
    image = load_image(image_address)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_location_list = face_recognition.face_locations(image_rgb)
    if face_location_list:
        face_location = face_location_list[0]
        cv2.rectangle(image, (face_location[3], face_location[0]), (face_location[1], face_location[2]),
                      (255, 0, 255), 2)
        image_faceLocation_dict[image_address] = (face_location[3], face_location[0], face_location[1], face_location[2])
        show_image(image)
    else:
        # face_detection_face_analyzer(image)
        coordinates = face_detection_haarcascade(image)
        if coordinates is None:
            body_coordinates = detect_person(image)
            coordinates = head_locate_from_body(image, body_coordinates)
        image_faceLocation_dict[image_address] = coordinates
        # print(f"image {i} coordinate info:", end=" ")
        # print(coordinates)


def face_recognition_solve_what():
    # Step 3:

    # with open("image_faceLocation_dict.pickle", "rb") as file:
    #     image_faceLocation_dict = pickle.load(file)
    #
    # for image_address, coordinates in image_faceLocation_dict.items():
    #     image = load_image(image_address)
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     encode_image_list = face_recognition.face_encodings(image_rgb)
    #     if encode_image_list:
    #         encode_image = encode_image_list[0]
    #     # If an image encoding list is not detected, it means the face is not detected
    #     else:
    #         coordinates_list = [(coordinates[1], coordinates[2], coordinates[3], coordinates[0])]
    #         encode_image = face_recognition.face_encodings(image_rgb, known_face_locations=coordinates_list)[0]
    #     image_faceEncoding_dict[image_address] = encode_image
    #
    # with open("image_faceEncoding_dict.pickle", "wb") as file:
    #     pickle.dump(image_faceEncoding_dict, file)

    face_comparison()


def face_comparison():
    with open("image_faceEncoding_dict.pickle", "rb") as file:
        image_faceEncoding_dict = pickle.load(file)

    ground_truth_list = list(image_faceEncoding_dict.values())
    final_evaluation_result = []
    for encode in ground_truth_list:
        results = face_recognition.compare_faces(ground_truth_list, encode)
        final_evaluation_result.append(results)
        print(results)

    evaluation_result_df = \
        pd.DataFrame(final_evaluation_result, columns=[image_address[10:-4] for image_address in list(image_faceEncoding_dict.keys())])
    evaluation_result_df.set_index(evaluation_result_df.columns, inplace=True)

    false_count = (~evaluation_result_df.values).sum()
    true_count = evaluation_result_df.values.sum()
    print(f"False count {false_count}, true count: {true_count}")

    mask = (~np.eye(len(evaluation_result_df), dtype=bool) & evaluation_result_df.values).any(axis=1)

    # Output the result for each row
    output = pd.Series(mask, index=evaluation_result_df.index)

    print(output)


    # encode1 = image_faceEncoding_dict["../images/image1_e.jpg"]
    #
    # encode2 = image_faceEncoding_dict["../images/image4_e_r.jpg"]
    #
    # ground_truth_list = [image_faceEncoding_dict["../images/image1_e.jpg"],
    #                      image_faceEncoding_dict["../images/image1_e_r.jpg"],
    #                      image_faceEncoding_dict["../images/image4_e.jpg"],
    #                      image_faceEncoding_dict["../images/image2_e.jpg"],
    #                      image_faceEncoding_dict["../images/image3_e.jpg"]]
    #
    # results = face_recognition.compare_faces(ground_truth_list, encode2)
    # print(results)









if __name__ == '__main__':
    # # Step 1: image preparation
    # image_path_list = []
    # for i in range(1, 7):
    #     large_image_address = f"../images/image{i}_e.jpg"
    #     image_path_list.append(large_image_address)
    #     # Uncomment this if the image folders do not contain a reduced version of image yet.
    #     # image_preparation(large_image_address)
    #
    # # Step 2: face detection, solve the where
    # for i in range(1, 7):
    #     if i == 3:
    #         continue
    #     small_image_address = f"../images/image{i}_e_r.jpg"
    #     image_path_list.append(small_image_address)
    #
    # image_path_list.append("../images/image7_e.jpg")
    #
    # for image_address in image_path_list:
    #     face_detection_solve_where(image_address)
    # # At the end of step 2, store the result about face location into
    # with open("image_faceLocation_dict.pickle", "wb") as file:
    #     pickle.dump(image_faceLocation_dict, file)

    # Step 3: face recognition, solve what
    face_recognition_solve_what()

    # face_detection_solve_where("../images/image7_e.jpg")
