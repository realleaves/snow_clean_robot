import cv2


def annotate(frame, regions):
    image = frame.color_image.copy()
    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
    return image
