import numpy as np
import cv2 as cv

def reproject(test_points,actual_img_pts,homo_mat,intrin):
    print('\n')
    for i in range(len(test_points)):
        # new_point=np.matmul(homo_mat,)
        # point_arr=np.array([[test_points[i][0]],[test_points[i][1]],[test_points[i][2]]])
        # print(point_arr)
        point_transformed = np.matmul(homo_mat, test_points[i])
        # point_transformed=point_transformed/point_transformed[2]
        inter_point = np.array([[point_transformed[0]], [point_transformed[1]], [point_transformed[2]]])
        # print(inter_point)
        final_point = np.transpose(np.matmul(intrin, inter_point))
        final_point_x = final_point[0][0]/final_point[0][2]
        final_point_y = final_point[0][1]/final_point[0][2]


        accuracy_x = final_point_x / actual_img_pts[i][0] * 100
        accuracy_y = final_point_y / actual_img_pts[i][1] * 100

        print('Reprojected : ', [final_point_x, final_point_y, 1], '       ' + 'Actual :', actual_img_pts[i],
              '         Accuracy = ', (accuracy_x + accuracy_y) / 2 )


world_pts = np.array([
 	[0,0,1],
    [22.5,0,1],
    [22.5,46.5,1],
    [0,46.5,1]
])

img_pts = np.array([
    [69,67,1],
    [28,301,1],
    [608,304,1],
    [570,73,1]
])



fx=384.352
fy=384.352
cx=321.095
cy=240.635

intrinsic = np.array([
	[fx,0,cx],
	[0,fy,cy],
	[0,0,1]])


intrinsic_inv = np.linalg.inv(intrinsic)
img_pts_prime = []

for point in img_pts:
  new_point = np.matmul(intrinsic_inv, point)
  img_pts_prime.append(new_point)


img_pts_prime = np.array(img_pts_prime)
# print(img_pts_prime)
homography_matrix, status = cv.findHomography(img_pts_prime, world_pts)
homography_matrix_normalised = homography_matrix/np.linalg.norm(homography_matrix)


rotation_matrix = np.zeros((3, 3))
r1 = homography_matrix_normalised[:, 0:1]
r2 = homography_matrix_normalised[:, 1:2]
r3 = np.transpose(np.cross(np.transpose(r1), np.transpose(r2)))

rotation_matrix[:, 0:1] = r1
rotation_matrix[:, 1:2] = r2
rotation_matrix[:, 2:3] = r3

translation_matrix = -(np.matmul(np.linalg.inv(rotation_matrix), homography_matrix_normalised[:, 2:3]))
print("ROTATION MATRIX :")
print(rotation_matrix)
print('\n')
print("TRANSLATION MATRIX :")
print(translation_matrix)
print('\n')
homography_matrix_inv = np.linalg.inv(homography_matrix)
reproject(world_pts, img_pts, homography_matrix_inv, intrinsic)