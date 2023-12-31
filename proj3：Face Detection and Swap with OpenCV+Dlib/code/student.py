import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(detector, predictor, img):
    '''
    This function first use `detector` to localize face bbox and then use `predictor` to detect landmarks (68 points, dtype: np.array).
    
    Inputs: 
        detector: a dlib face detector
        predictor: a dlib landmark detector, require the input as face detected by detector
        img: input image
        
    Outputs:
        landmarks: 68 detected landmark points, dtype: np.array

    '''

    # TODO: Implement this function!
    img_copy = img.copy()
    faces = detector(img_copy, 1)

    if len(faces) > 1:
        raise TooManyFaces
    if len(faces) == 0:
        raise NoFaces

    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_copy, faces[0]).parts()])

    return landmarks


def get_face_mask(img, landmarks):
    '''
    This function gets the face mask according to landmarks.
    
    Inputs: 
        img: input image
        landmarks: 68 detected landmark points, dtype: np.array
        
    Outputs:
        convexhull: face convexhull
        mask: face mask 

    '''

    # TODO: Implement this function!
    img_copy = img.copy()

    convexhull = cv2.convexHull(landmarks)  # Find the convex hull

    mask = np.zeros(img_copy.shape[:2], dtype=np.float64)
    cv2.fillConvexPoly(mask, convexhull, color=1)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    # gauss blur with a blur radius of 7 to make the edges smoother
    mask = (cv2.GaussianBlur(mask, (7, 7), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = mask.astype(np.uint8)

    return convexhull, mask


def get_delaunay_triangulation(landmarks, convexhull):
    '''
    This function gets the face mesh triangulation according to landmarks.
    
    Inputs: 
        landmarks: 68 detected landmark points, dtype: np.array
        convexhull: face convexhull
        
    Outputs:
        triangles: face triangles 
    '''

    # TODO: Implement this function!
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks.tolist())
    triangles = subdiv.getTriangleList()
    return triangles


def transformation_from_landmarks(target_landmarks, source_landmarks):
    '''
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    
    Inputs: 
        target_landmarks: 68 detected landmark points of the target face, dtype: np.array
        source_landmarks: 68 detected landmark points of the source face that need to be warped, dtype: np.array
        
    Outputs:
        triangles: face triangles 
    '''
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    # TODO: Implement this function!
    points1 = target_landmarks.astype(np.float64)
    points2 = source_landmarks.astype(np.float64)
    # Eliminate the effect of translational T
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    # Eliminate the effect of scaling factor S
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    M = np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    return M


def warp_img(img, M, target_shape):
    '''
    This function utilizes the affine transformation matrix M to transform the img.
    
    Inputs: 
        img: input image (np.array) need to be warped.
        M: affine transformation matrix.
        target_shape: the image shape of target image
        
    Outputs:
        warped_img: warped image.
    
    '''

    # TODO: Implement this function!
    img_copy = img.copy()
    warped_img = np.zeros(target_shape, dtype=img_copy.dtype)
    cv2.warpAffine(img_copy, M[:2], (target_shape[1], target_shape[0]), dst=warped_img, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return warped_img


def correct_colours(img1, img2, landmarks1):
    """
    This function corrects the color of the warped image.
    """
    COLOUR_CORRECT_BLUR_FRAC = 0.9
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = 128 * (im2_blur <= 1.0) + im2_blur

    return img2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)


def show_landmarks(img1, landmarks1):
    '''
    This function shows the image with landmarks.

    Inputs:
        img1: input image (np.array) need to be warped.
        landmarks1: 68 detected landmark points of the target face, dtype: np.array

    Outputs:
        None
    '''
    img1_copy = img1.copy()
    for i in range(68):
        x = landmarks1[i, 0]
        y = landmarks1[i, 1]
        cv2.circle(img1_copy, (x, y), 2, (0, 255, 0), -1)
    plt.imshow(img1_copy)
    plt.show()
