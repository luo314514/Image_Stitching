import cv2
img = cv2.imread("building_A.jpg")
print("图片的真实宽高为: 宽度 =", img.shape[1], ", 高度 =", img.shape[0])