import time
from collections import defaultdict
from typing import Dict

import cv2
import numpy as np
import supervision as sv

CLASSES = [
    "black boba bowl",
    "purple cup",
    "green cup",
    "red cup",
    "milk cup",
    "cup stack",
]

CLASS_TO_DESCRIPTION = {
    "black boba bowl": "bowl with boba",
    "purple cup": "cup with taro",
    "green cup": "cup with matcha",
    "red cup": "cup with strawberry",
    "milk cup": "cup with milk",
    "cup stack": "cup stack",
}
SCENE_KEYS = tuple(
    list(CLASS_TO_DESCRIPTION.values())
    + [
        "empty cup",
        "finished location",
        "trash location",
    ]
)
CLASS_TO_LABEL = {
    "black boba bowl": "boba",
    "purple cup": "taro",
    "green cup": "matcha",
    "red cup": "strawberry",
    "milk cup": "milk",
    "cup stack": "cup stack",
}

CLASSES_len = len(CLASSES)


def getxiyi(detections: sv.Detections) -> Dict[int, np.ndarray]:
    xyxy_map = defaultdict(list)
    for _id, xyxy in zip(detections.class_id, detections.xyxy):
        xyxy_map[_id].append(xyxy)

    # choose the center of the largest area for each index
    center = {}
    for k, v in xyxy_map.items():
        maxarea = 0
        index = 0
        for i in range(len(v)):
            xyxy = v[i]
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area > maxarea:
                maxarea = area
                index = i
        xyxy = v[index]
        _x = (xyxy[0] + xyxy[2]) / 2.0
        _y = xyxy[3]
        center[k] = [_x, _y]
    return center


# click event function
class ClickStore:
    def __init__(self) -> None:
        self._clicks = []

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, ",", y)
            self._clicks.append((x, y))

    def get_clicks(self):
        return self._clicks


def click_location(image):
    xi, yi = [], []
    for CLASS in CLASSES:
        get_clicks = []
        while len(get_clicks) == 0:
            print("please click the location of the " + CLASS)
            cv2.imshow("image", image[:, :, ::-1])
            cv2.moveWindow("image", 50, 50)
            # calling the mouse click event
            store = ClickStore()
            cv2.setMouseCallback("image", store.click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            get_clicks = store.get_clicks()

        x = np.mean([get_clicks[i][0] for i in range(len(get_clicks))])
        y = np.mean([get_clicks[i][1] for i in range(len(get_clicks))])
        xi.append(x)
        yi.append(y)
    return xi, yi


def getkb(x, yi):
    if (yi[1] - yi[0]) == 0:
        print("error! yi[1] == yi[0], please provide different yi[1] and yi[0]")
        exit()
    else:
        k = (x[1] - x[0]) / (yi[1] - yi[0])
        b = x[2] - k * yi[2]
    return k, b


def get_location(image, model, kx, bx, ky, by, show_and_wait_for_confirm: bool = False):
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3

    start_time = time.time()
    detections = model.predict_with_classes(
        image=image[:, :, ::-1],  # expects BGR (look at documentation of this function)
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    print(f"detection took: {time.time() - start_time}")
    centers = getxiyi(detections)
    if show_and_wait_for_confirm:
        print(centers)
        print(CLASSES)
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        cv2.imshow("annotated_image", annotated_image[:, :, ::-1])
        cv2.moveWindow("annotated_image", 50, 50)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    # get the center points
    if not set(range(CLASSES_len)) == set(centers.keys()):
        print("*** not all the CLASSES are detected ***")
        # interface to ask for human to give annotation of where the cup is.
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        # Image.fromarray(annotated_image[:,:,::-1])
        xi, yi = click_location(annotated_image)
    else:
        xi, yi = [], []
        for CLASS in range(len(CLASSES)):
            x, y = centers[CLASS]
            xi.append(x)
            yi.append(y)

    x = [(kx * yi_ + bx) / 100 for yi_ in yi]
    y = [(ky * xi_ + by) / 100 for xi_ in xi]
    scene = {}

    for CLASSESi in range(CLASSES_len):
        class_description = CLASS_TO_DESCRIPTION[CLASSES[CLASSESi]]
        scene[class_description] = (x[CLASSESi], y[CLASSESi])

    if show_and_wait_for_confirm:
        print(scene)
        print(xi, yi)
        # show the image
        img_show = image.copy()
        for _class, _x, _y in zip(CLASSES, xi, yi):
            class_name = CLASS_TO_LABEL[_class]
            img_show = cv2.putText(
                img_show,
                class_name,
                (int(_x), int(_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # put a dot on the object
            img_show = cv2.circle(
                img_show,
                (int(_x), int(_y)),
                5,
                (255, 0, 0),
                -1,
            )

        cv2.imshow("image", img_show[:, :, ::-1])
        cv2.moveWindow("image", 50, 50)

        # wait for a key to be pressed to exit
        cv2.waitKey(3000)
        # close the window
        cv2.destroyAllWindows()

    # truncated each location to 3 decimal places
    for key in scene.keys():
        scene[key] = tuple([round(i, 3) for i in scene[key]])
    return scene
