from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import correlate

def load_images(directory):
    """We load the 64 images using the skimage.io.ImageCollection() function,
    and save the images as a skimage.io.ImageCollection object. This object
    is then cast as a 3D numpy matrix."""

    # Load collection of images from given directory
    image_collection = io.ImageCollection(directory + '/*.png', as_gray=True)
    # Stack images to generate 3D matrix where one axis represents time
    images_matrix = np.stack(image_collection)

    return images_matrix


def show_animation(image_matrix, delay=10, title=""):

    """We use the matplotlib.animation.FuncAnimation() function to
    show a series of images given as a 3D Matrix  (where the first axis
    represents time) as an animation. """

    def update_frame(frame):

        """Shows an image"""

        ax.clear()
        ax.imshow(image_matrix[frame,:,:], cmap='gray', vmax=1, vmin=0)
        ax.set_title(f'{title} - Frame #{frame}')
        ax.axis('off')

    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update_frame, frames=image_matrix.shape[0], interval=delay)
    plt.show()


def filter_images(image_matrix, kernel, axis=0):

    imgs = image_matrix.copy()

    if axis == 0:
        for t in range(imgs.shape[0]):
            imgs[t,:,:] = correlate(image_matrix[t,:,:], kernel)
    elif axis == 1:
        for y in range(imgs.shape[1]):
            imgs[:, y, :] = correlate(image_matrix[:, y, :], kernel)
    elif axis == 2:
        for x in range(imgs.shape[2]):
            imgs[:, :, x] = correlate(image_matrix[:, :, x], kernel)
    else:
        return None

    return imgs




directory = 'toyProblem_F22'
V = load_images(directory)
N_t, N_y, N_x = V.shape

V_t, V_y, V_x = [np.zeros((N_t-1, N_y-1, N_x-1), dtype=np.float32) for _ in range(3)]

# Problem 2.1

V_t = V[1:, 1:, 1:] - V[:-1, 1:, 1:]
V_y = V[1:, 1:, 1:] - V[1:, :-1, 1:]
V_x = V[1:, 1:, 1:] - V[1:, 1:, :-1]

# Alternative method:
# for t in range(1, N_t):
#     for x in range(1, N_y):
#         for y in range(1, N_x):
#             V_t[t - 1, y - 1, x - 1] = V[t, y, x] - V[t - 1, y, x]
#             V_y[t - 1, y - 1, x - 1] = V[t, y, x] - V[t, y - 1, x]
#             V_x[t - 1, y - 1, x - 1] = V[t, y, x] - V[t, y, x - 1]

# # Frames with no motion
# io.imshow(V_t[26,:,:])
# io.imshow(V_t[53,:,:])


# show_animation(V, title="V")
# show_animation(V_x, title=r"$V_x$")
# show_animation(V_y, title=r"$V_y$")
# show_animation(V_t, title=r"$V_t$")

# # Each dimension is reduced by 1 because we need one
# # previous step at i-1 to calculate the gradient at i.
# print(V.shape)
# print(V_x.shape, V_y.shape, V_t.shape)
#
#
# # Filter kernels

prewitt_x = [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1]]

prewitt_y = [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]]

sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

sobel_y = [[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]]

# V_prewitt_x = filter_images(V, prewitt_x, axis=0)
# V_prewitt_y = filter_images(V, prewitt_y, axis=0)
# V_sobel_x = filter_images(V, sobel_x, axis=0)
# V_sobel_y = filter_images(V, sobel_y, axis=0)
# V_t_prewitt = filter_images(V, prewitt_y, axis=2)
# V_t_sobel = filter_images(V, sobel_y, axis=2)
#
# show_animation(V_prewitt_x, title="Horizontal Prewitt filter")
# show_animation(V_prewitt_y, title="Vertical Prewitt filter")
# show_animation(V_sobel_x, title="Horizontal Sobel filter")
# show_animation(V_sobel_y, title="Vertical Sobel filter")
# show_animation(V_t_prewitt, title="Vertical Prewitt filter aplied to time axis")
# show_animation(V_t_sobel, title="Vertical Sobel filter applied to time axis")

def one_dimensional_gaussian_filter(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - (size - 1) / 2) ** 2) / (2 * sigma ** 2)), (size,1))
    return kernel / np.sum(kernel)

def two_dimensional_gaussian_filter(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


sigma = 1
size = 6*sigma

G1 = one_dimensional_gaussian_filter(size, sigma)


V_gaussian1 = filter_images(V, G1, axis=0)
V_gaussian1 = filter_images(V_gaussian1, np.transpose(G1), axis=0)
show_animation(V_gaussian1, title=f"1D Gaussian filter of length {size} applied twice")

G2 = two_dimensional_gaussian_filter(size, sigma)
V_gaussian2 = filter_images(V, G2, axis=0)
show_animation(V_gaussian2, title=f"2D Gaussian filter of dimension {size}X{size}")

# Difference of the two methods (black indicates no difference)
show_animation(np.abs(V_gaussian1-V_gaussian2), title="Difference between applying 1D Gauss twice and 2D Gauss once")

# 3D Gaussian filter:

G = one_dimensional_gaussian_filter(size, sigma)

V_gaussian = filter_images(V, G, axis=0)
V_gaussian = filter_images(V_gaussian, np.transpose(G), axis=0)
V_gaussian = filter_images(V_gaussian, G, axis=2)

show_animation(V_gaussian)