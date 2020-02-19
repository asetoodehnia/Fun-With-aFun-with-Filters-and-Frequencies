import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.color import rgb2gray

from scipy import signal
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import cv2
import copy

from align_image_code import align_images

D_x = np.asarray([[1, -1]])
D_y = np.asarray([[1], [-1]])

root = "img/"

def main():
	# part_1()
	part_2()

def part_1():
	# Part 1.1
	cameraman = skio.imread(root + "cameraman.png", as_gray=True)
	cameraman = sk.img_as_float(cameraman)

	result_D_x = signal.convolve2d(cameraman, D_x, mode='same')
	result_D_y = signal.convolve2d(cameraman, D_y, mode='same')

	skio.imsave("part_1/1_1/cameraman_D_x.png", result_D_x)
	skio.imsave("part_1/1_1/cameraman_D_y.png", result_D_y)

	grad_img = np.sqrt(result_D_x**2 + result_D_y**2)
	skio.imsave("part_1/1_1/cameraman_grad_mag.png", grad_img)

	threshold = 0.15
	grad_img_filtered = np.where(grad_img > threshold, 1, 0)
	skio.imsave("part_1/1_1/cameraman_grad_thresh.png", grad_img_filtered)

	# Part 1.2
	gaussian = cv2.getGaussianKernel(12, 0)
	gaussian = gaussian@gaussian.T
	# skio.imsave("part_1/1_2/gaussian_kernel_1_2.png", gaussian)

	blurred_cameraman = signal.convolve2d(cameraman, gaussian, mode='same')
	skio.imsave("part_1/1_2/gaussian_cameraman.png", blurred_cameraman)

	blurred_result_D_x = signal.convolve2d(blurred_cameraman, D_x, mode='same')
	blurred_result_D_y = signal.convolve2d(blurred_cameraman, D_y, mode='same')
	blurred_grad_img = np.sqrt(blurred_result_D_x**2 + blurred_result_D_y**2)

	skio.imsave("part_1/1_2/gaussian_cameraman_D_x.png", blurred_result_D_x)
	skio.imsave("part_1/1_2/gaussian_cameraman_D_y.png", blurred_result_D_y)
	skio.imsave("part_1/1_2/gaussian_cameraman_grad_mag.png", blurred_grad_img)

	threshold = 0.05
	blurred_grad_img_filtered = np.where(blurred_grad_img > threshold, 1, 0)
	skio.imsave("part_1/1_2/gaussian_cameraman_grad_thresh.png", blurred_grad_img_filtered)

	gaussian_D_x = signal.convolve2d(gaussian, D_x, mode='same')
	gaussian_D_y = signal.convolve2d(gaussian, D_y, mode='same')

	# skio.imsave("part_1/1_2/gaussian_D_x.png", gaussian_D_x)
	# skio.imsave("part_1/1_2/gaussian_D_y.png", gaussian_D_y)

	DoG_result_D_x = signal.convolve2d(cameraman, gaussian_D_x, mode='same')
	DoG_result_D_y = signal.convolve2d(cameraman, gaussian_D_y, mode='same')
	DoG_grad_img = np.sqrt(DoG_result_D_x**2 + DoG_result_D_y**2)

	skio.imsave("part_1/1_2/DoG_cameraman_D_x.png", DoG_result_D_x)
	skio.imsave("part_1/1_2/DoG_cameraman_D_y.png", DoG_result_D_y)
	skio.imsave("part_1/1_2/DoG_cameraman_grad_mag.png", DoG_grad_img)

	threshold = 0.05
	DoG_grad_img_filtered = np.where(DoG_grad_img > threshold, 1, 0)
	skio.imsave("part_1/1_2/DoG_cameraman_grad_thresh.png", DoG_grad_img_filtered)
	return
	# 1.3
	pics = ["facade.jpg", "me_desert.jpg", "me_diner.jpg", "me_NYSE.jpg", "me_redwoods.jpg"]
	ranges = [np.arange(-5, 6, 1), np.arange(0, 15, 1), np.arange(-5, 6, 1), 
			  np.arange(-25, -15, 1), np.arange(-5, 6, 1)]
	i = 0
	for pic in pics:
		img = skio.imread(root + pic)
		img_gray = rgb2gray(img)

		angle, angle_distr, original_distr = find_rotation_angle(img_gray, ranges[i], gaussian)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(angle_distr, bins=180)
		ax.set_title(str(angle) + " degrees rotation histogram for " + pic)
		fig.savefig("part_1/1_3/" + str(angle) + "_hist_" + pic)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(original_distr, bins=180)
		ax.set_title("0 degrees rotation histogram for " + pic)
		fig.savefig("part_1/1_3/original_hist_" + pic)

		result = rotate(img, angle)
		skio.imsave("part_1/1_3/result_" + pic, result)
		
		i += 1

def part_2():
	# # 2.1
	# gaussian = cv2.getGaussianKernel(12, 6)
	# gaussian = gaussian@gaussian.T

	# taj = skio.imread(root + 'taj.jpg')
	# taj = sk.img_as_float(taj)
	# taj_sharp = np.clip(unsharp_mask_filter(taj, 1, gaussian), 0, 1)
	# skio.imsave("part_2/2_1/taj_sharp.jpg", taj_sharp)

	# nyc = skio.imread(root + 'nyc.jpg')
	# nyc = sk.img_as_float(nyc)
	# nyc_sharp = np.clip(unsharp_mask_filter(nyc, 1, gaussian), 0, 1)
	# skio.imsave("part_2/2_1/nyc_sharp.jpg", nyc_sharp)

	# sugar_bowl = skio.imread(root + 'sugar_bowl.jpg')
	# sugar_bowl = sk.img_as_float(sugar_bowl)
	# sugar_bowl_blurred = convolve_rgb(sugar_bowl, gaussian)
	# sugar_bowl_sharp = np.clip(unsharp_mask_filter(sugar_bowl_blurred, 1, gaussian), 0, 1)
	# skio.imsave("part_2/2_1/sugar_bowl_blurred.jpg", sugar_bowl_blurred)
	# skio.imsave("part_2/2_1/sugar_bowl_sharp.jpg", sugar_bowl_sharp)

	# # 2.2
	# im1 = skio.imread('img/DerekPicture.jpg') / 255
	# im2 = skio.imread('img/nutmeg.jpg') / 255

	# hybrid, low, high = hybrid_image(im1, im2, 20, 12, 20, 12)
	# skio.imshow(hybrid, cmap='gray')
	# plt.show()
	# skio.imsave('part_2/2_2/derek_nutmeg.png', hybrid)

	# im1 = skio.imread('img/messi.png') / 255
	# im2 = skio.imread('img/goat.jpg') / 255

	# hybrid, low, high = hybrid_image(im1, im2, 20, 5, 20, 2)
	# skio.imshow(hybrid, cmap='gray')
	# plt.show()
	# skio.imsave('part_2/2_2/messi_goat.png', hybrid)

	# low_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(low))))
	# skio.imshow(low_fft)
	# plt.show()
	# skio.imsave('part_2/2_2/messi_low_fft.jpg', low_fft, cmap='gray')

	# high_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(high))))
	# plt.imshow(high_fft)
	# plt.show()
	# skio.imsave('part_2/2_2/goat_high_fft.jpg', high_fft, cmap='gray')

	# im1 = skio.imread('img/zlatan.jpg') / 255
	# im2 = skio.imread('img/lion.jpg') / 255

	# hybrid, low, high = hybrid_image(im1, im2, 20, 5, 20, 2)
	# skio.imshow(hybrid, cmap='gray')
	# plt.show()
	# skio.imsave('part_2/2_2/zlatan_lion.jpg', hybrid)

	# low_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(low))))
	# skio.imshow(low_fft)
	# plt.show()
	# skio.imsave('part_2/2_2/zlatan_low_fft.jpg', low_fft, cmap='gray')

	# high_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(high))))
	# plt.imshow(high_fft)
	# plt.show()
	# skio.imsave('part_2/2_2/lion_high_fft.jpg', high_fft, cmap='gray')

	# # 2.3
	# dali = skio.imread("img/lincoln.jpg")
	# stack_g, stack_l = get_stacks(dali, 5, cv2.getGaussianKernel(20, 5))
	# for i in range(len(stack_g)):
	# 	skio.imsave("part_2/2_3/dali_gauss_"+ str(i) + ".jpg", stack_g[i])
	# 	skio.imsave("part_2/2_3/dali_laplace_"+ str(i+1) + ".jpg", stack_l[i])

	# face = skio.imread("img/face.jpg")
	# stack_g, stack_l = get_stacks(face, 5, cv2.getGaussianKernel(20, 5))
	# for i in range(len(stack_g)):
	# 	skio.imsave("part_2/2_3/face_gauss_"+ str(i) + ".jpg", stack_g[i])
	# 	skio.imsave("part_2/2_3/face_laplace_"+ str(i+1) + ".jpg", stack_l[i])

	# messi_goat = skio.imread("part_2/2_2/messi_goat.png")
	# stack_g, stack_l = get_stacks(messi_goat, 5, cv2.getGaussianKernel(20, 5), True)
	# for i in range(len(stack_g)):
	# 	skio.imsave("part_2/2_3/messi_goat_gauss_"+ str(i) + ".jpg", stack_g[i])
	# 	skio.imsave("part_2/2_3/messi_goat_laplace_"+ str(i+1) + ".jpg", stack_l[i])

	# zlatan_lion = skio.imread("part_2/2_2/zlatan_lion.jpg")
	# stack_g, stack_l = get_stacks(zlatan_lion, 5, cv2.getGaussianKernel(20, 5), True)
	# for i in range(len(stack_g)):
	# 	skio.imsave("part_2/2_3/zlatan_lion_gauss_"+ str(i) + ".jpg", stack_g[i])
	# 	skio.imsave("part_2/2_3/zlatan_lion_laplace_"+ str(i+1) + ".jpg", stack_l[i])

	# 2.4
	apple = skio.imread("img/apple.jpeg")
	apple = sk.img_as_float(apple)
	orange = skio.imread("img/orange.jpeg")
	orange = sk.img_as_float(orange)

	mask = np.zeros_like(apple)
	h, w, d = apple.shape
	for i in range(w // 2):
	    mask[:, i] = np.ones((h, 1))
	    
	result, levels = blend_imgs(apple, orange, mask, 20, cv2.getGaussianKernel(30, 12), 
												 cv2.getGaussianKernel(30, 12), 
												 cv2.getGaussianKernel(30, 12))

	skio.imsave("part_2/2_4/orapple.jpg", result)
	for i in range(len(levels)):
		skio.imsave("part_2/2_4/level_" + str(i) + "_orapple.jpg", levels[i])
	return
	nyc_day = skio.imread("img/nyc_day.png")
	nyc_day = sk.img_as_float(nyc_day)
	nyc_day = nyc_day[:,:,:3]
	nyc_night = skio.imread("img/nyc_night.png")
	nyc_night = sk.img_as_float(nyc_night)
	nyc_night = nyc_night[:,:,:3]

	h, w, d = nyc_day.shape

	mask = np.zeros_like(nyc_day)
	for i in range(w // 2):
	    mask[:, i] = np.ones((h, 1))
	    
	result, levels = blend_imgs(nyc_day, nyc_night, mask, 20, cv2.getGaussianKernel(30, 12), 
													  cv2.getGaussianKernel(30, 12), 
													  cv2.getGaussianKernel(30, 12))

	skio.imsave("part_2/2_4/nyc_blended.jpg", result)

	me = skio.imread("img/me_for_matt.jpg")
	me = sk.img_as_float(me)
	matt = skio.imread("img/matt_for_me.jpg")
	matt = sk.img_as_float(matt)

	mask = np.zeros_like(me)
	h, w, d = me.shape
	for i in range(w // 2):
		mask[:, i] = np.ones((h, 1))

	result, levels = blend_imgs(me, matt, mask, 12, cv2.getGaussianKernel(12, 10), 
	                                        cv2.getGaussianKernel(12, 10), 
	                                        cv2.getGaussianKernel(12, 10))

	skio.imsave("part_2/2_4/me_and_matt.jpg", result)



############################
##### HELPER FUNCTIONS #####
############################

def blend_imgs(im1, im2, mask, n, gaussian1, gaussian2, gaussian3):
	_, im1_lap_stack = get_stacks(im1, n, gaussian1)
	_, im2_lap_stack = get_stacks(im2, n, gaussian2)
	mask_gauss_stack, _ = get_stacks(mask, n, gaussian3)

	levels = [mask_gauss_stack[l]*im1_lap_stack[l] + (1 - mask_gauss_stack[l])*im2_lap_stack[l] for l in range(n)]
	collapsed = np.sum(levels, axis=0)
	return collapsed, levels

def get_stacks(img, n, gaussian, gray=False):
    gauss_stack = []
    lap_stack = []
    curr_gauss = copy.copy(img)
    for i in range(1, n):
        curr_gauss = curr_gauss
        gauss_stack.append(curr_gauss)
        next_gauss = low_pass(curr_gauss, gaussian, gray)
        lap_image = curr_gauss - next_gauss
        lap_stack.append(lap_image)
        curr_gauss = next_gauss
    gauss_stack.append(curr_gauss)
    lap_stack.append(curr_gauss)

    return gauss_stack, lap_stack

def normalize(array):
	return (array - np.min(array)) / (np.max(array) - np.min(array))

def hybrid_image(im1, im2, k1, sigma1, k2, sigma2):
	# Next align images (this code is provided, but may be improved)
	im2_aligned, im1_aligned = align_images(im2, im1)

	im1_aligned = rgb2gray(im1_aligned)
	im2_aligned = rgb2gray(im2_aligned)

	low_pass_im = low_pass(im1_aligned, cv2.getGaussianKernel(k1, sigma1), True)
	high_pass_im = high_pass(im2_aligned, cv2.getGaussianKernel(k2, sigma2), True)
	return low_pass_im + high_pass_im, low_pass_im, high_pass_im

def low_pass(img, gaussian, gray=False):
    gaussian = gaussian@gaussian.T
    if gray:
        return signal.convolve2d(img, gaussian, mode='same')
    return convolve_rgb(img, gaussian)

def high_pass(img, gaussian, gray=False):
	low_pass_img = low_pass(img, gaussian, gray)
	return np.subtract(img, low_pass_img)

def convolve_rgb(img, conv_filter, flag=False):
    """
    Convolves im with conv_filter, over all three colour channels
    """
    channels = []
    for i in range(3):
        channels.append(signal.convolve2d(img[:,:,i], conv_filter, mode="same"))
    result = np.stack(channels, axis=2)
    if flag:
        return result
    return result
	# if flag:
	# 	return result
	
	# return normalize(result)

def unsharp_mask_filter(img, alpha, gaussian):
	e = np.zeros_like(gaussian)
	e[gaussian.shape[0]//2 + 1][gaussian.shape[0]//2 + 1] = 1
	conv_filter = (1 + alpha)*e - alpha*gaussian

	return convolve_rgb(img, conv_filter)


def find_rotation_angle(image, angle_range, gaussian):
	'''
	takes in a grayscale image, and finds the ideal angle of rotation based on 
	number of vertical and horizontal edges found using a gaussian derivative filter
	'''
	D_x = np.asarray([[1, -1]])
	D_y = np.asarray([[1], [-1]])

	indices = [0, 1, 89, 90, 91, 179, 180]

	best_angle = 0
	best_score = -float('inf')
	angles_distr = None
	angle_range = np.append(angle_range, 0)

	for i in angle_range:
		rotated = rotate(image, i)
		rotated = rotated[int(0.2 * len(rotated)) : -int(0.2 * len(rotated)), 
						  int(0.2 * len(rotated[0])) : -int(0.2 * len(rotated[0]))]

		blurred_img = signal.convolve2d(rotated, gaussian, mode='same')

		img_D_x = signal.convolve2d(blurred_img, D_x, mode='same')
		img_D_y = signal.convolve2d(blurred_img, D_y, mode='same')

		angles = np.ndarray.flatten(np.abs(np.arctan2(-img_D_y, img_D_x) * 180 / np.pi))
		angle_hist = np.histogram(angles, bins=np.arange(0, 181, 1))

		curr_score = 0
		for idx in indices:
			try:
				curr_score += angle_hist[0][idx]
			except:
				continue

		if curr_score > best_score:
			best_score = curr_score
			best_angle = i
			angles_distr = angles
		if i == 0:
			original_distr = angles

	return best_angle, angles_distr, original_distr

if __name__ == '__main__':
	main()










