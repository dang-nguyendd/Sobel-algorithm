"""
@file sobel_demo.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import sys
import cv2 as cv


def main():
    ## [variables]
    # First we declare the variables we are going to use
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 0.1
    delta = 2
    ddepth = cv.CV_16S
    ## [variables]

    ## [load]
    # As usual we load our source image (src)
    # Check number of arguments

    src = cv.imread("test_final_perspective.png")
    src = cv.resize(src, (800, 800))

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ')
        return -1
    ## [load]

    ## [reduce_noise]
    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    src = cv.GaussianBlur(src, (3, 3), 0)
    ## [reduce_noise]

    ## [convert_to_gray]
    # Convert the image to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [sobel]
    # Gradient-X
    # grad_x = cv.Scharr(gray,ddepth,1,0)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    ## [sobel]

    ## [convert]
    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    ## [convert]

    ## [blend]
    ## Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ## [blend]

    ## [exclude_edges]
    # Exclude edges from the original image by setting them to a background color
    background_color = (255, 255, 255)  # White in BGR format
    result_image = src.copy()
    result_image[grad > 50] = background_color  # Adjust the threshold as needed
    ## [exclude_edges]

    ## [display_result]
    # Display the original and result images
    cv.imshow('Original Image', src)
    cv.imshow('Result Image', result_image)
    cv.waitKey(0)
    ## [display_result]

    ## [export_image]
    # Export the resulting image to a PNG file
    cv.imwrite('C:/Users/Admin/PycharmProjects/sobelProject/final_sobel_perspective.png', result_image)
    ## [export_image]

    return 0

if __name__ == "__main__":
    main()