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
        ax.imshow(image_matrix[frame, :, :], cmap='gray', vmax=1, vmin=0)
        ax.set_title(f'{title} - frame #{frame}')
        ax.axis('off')

    fig, ax = plt.subplots()
    Animation = FuncAnimation(fig, update_frame, frames=image_matrix.shape[0], interval=delay)
    plt.show()


def normalize_images(X):
    """This function normalizes the values to [0,1] of each of the images in the image collection."""

    for t in range(X.shape[0]):
        min_val, max_val = np.min(X[t, :, :]), np.max(X[t, :, :])
        if max_val > min_val:
            X[t, :, :] = (X[t, :, :] - min_val) / (max_val - min_val)
        else:
            X[t, :, :] = np.zeros_like(X[t, :, :])


def filter_images(image_matrix, kernel, axis=0):
    """This function uses the correlate funciton from scipy.ndimage.
    Depending on the axis \in {0,1,2} that is chosen, the correlate function is
    applied to each "layer" of the 3D matrix differently."""

    imgs = image_matrix.copy()

    if axis == 0:
        for t in range(imgs.shape[0]):
            imgs[t, :, :] = correlate(image_matrix[t, :, :], kernel)
        normalize_images(imgs)

    elif axis == 1:
        for y in range(imgs.shape[2]):
            imgs[:, y, :] = correlate(image_matrix[:, y, :], kernel)
        normalize_images(imgs)

    elif axis == 2:
        for x in range(imgs.shape[2]):
            imgs[:, :, x] = correlate(image_matrix[:, :, x], kernel)
        normalize_images(imgs)

    else:
        return None

    return imgs


# --------------------------------------------------- PROBLEM 1

directory = 'toyProblem_F22'
V = load_images(directory)
N_t, N_y, N_x = V.shape

show_animation(V, title="All images")

# -------------------------------------------------------------





# # --------------------------------------------------- PROBLEM 2.1

V_t, V_y, V_x = [np.zeros((N_t - 1, N_y - 1, N_x - 1), dtype=np.float32) for _ in range(3)]

# It is quite easy to claculate the gradients as described the description for problem 2.1.
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

normalize_images(V_x)
normalize_images(V_y)
normalize_images(V_t)

# # Frames with no motion
# io.imshow(V_t[26,:,:])
# io.imshow(V_t[53,:,:])

show_animation(V_x, title=r"$V_x$")
show_animation(V_y, title=r"$V_y$")
show_animation(V_t, title=r"$V_t$")

# Each dimension is reduced by 1 because we need one
# previous step at i-1 to calculate the gradient at i.

# -------------------------------------------------------------





# --------------------------------------------------- PROBLEM 2.2

# Filter kernels

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

V_prewitt_x = filter_images(V, prewitt_x, axis=0)
V_prewitt_y = filter_images(V, prewitt_y, axis=0)
V_sobel_x = filter_images(V, sobel_x, axis=0)
V_sobel_y = filter_images(V, sobel_y, axis=0)
V_prewitt_t = filter_images(V, prewitt_y, axis=2)
V_sobel_t = filter_images(V, sobel_y, axis=2)

show_animation(V_prewitt_x, title="Horizontal Prewitt filter")
show_animation(V_prewitt_y, title="Vertical Prewitt filter")
show_animation(V_sobel_x, title="Horizontal Sobel filter")
show_animation(V_sobel_y, title="Vertical Sobel filter")
show_animation(V_prewitt_t, title="Vertical Prewitt filter aplied to time axis")
show_animation(V_sobel_t, title="Vertical Sobel filter applied to time axis")


# Plotting example images and comparing with results from 2.1

fig, axs = plt.subplots(4, 3, figsize=(30, 10))

def plot_image(X, ax, title):
    ax.imshow(X, cmap="gray")
    ax.set_title(title)
    ax.axis('off')

plot_image(V_prewitt_x[2], axs[0, 0], "$V_x$ calculated with Prewitt filter")
plot_image(V_x[2], axs[1, 0], "$V_x$ calculated manually")

plot_image(V_prewitt_y[2], axs[0, 1], "$V_y$ calculated with Prewitt filter")
plot_image(V_y[2], axs[1, 1], "$V_y$ calculated manually")

plot_image(V_prewitt_t[2], axs[0, 2], "$V_t$ calculated with Prewitt filter")
plot_image(V_t[2], axs[1, 2], "$V_t$ calculated manually")

plot_image(V_sobel_x[2], axs[2, 0], "$V_x$ calculated with Sobel filter")
plot_image(V_x[2], axs[3, 0], "$V_x$ calculated manually")

plot_image(V_sobel_y[2], axs[2, 1], "$V_y$ calculated with Sobel filter")
plot_image(V_y[2], axs[3, 1], "$V_y$ calculated manually")

plot_image(V_sobel_t[2], axs[2, 2], "$V_t$ calculated with Sobel filter")
plot_image(V_t[2], axs[3, 2], "$V_t$ calculated manually")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------




# --------------------------------------------------- PROBLEM 2.3

# We can create 1D Gaussian kernels with this function
def one_dimensional_gaussian_filter(size, sigma=1.0):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-(x / (2 * sigma)) ** 2)
    return [list(kernel)]

sigma = 1/2
size = int(6 * sigma)

# We will show that we get the same result by applying the 1D filter twice
# as we do by applying the 2D filter once.

kernel1D = one_dimensional_gaussian_filter(size, sigma)
kernel2D = np.transpose(kernel1D) @ kernel1D

V_blurred = filter_images(V, kernel1D, axis=0)
V_blurred1 = filter_images(V_blurred, np.transpose(kernel1D), axis=0)

show_animation(V_blurred, title="V filtered with 1D gaussian twice ($G(x)$ and then $G(y)$)")

V_blurred2 = filter_images(V, kernel2D, axis=0)

show_animation(V_blurred, title="V filtered with 2D gaussian once ($G(x)G(y)$)")

# There is no difference

difference = abs(V_blurred1 - V_blurred2)
show_animation(difference, title="Difference between images blurred using the two methods")
print(np.mean(difference), np.min(difference), np.max(difference))


def one_dimensional_derivative_gaussian_filter(size, sigma=1.0):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = -(np.sqrt(2) * x * np.exp(-(x ** 2) / (2 * sigma ** 2))) / (2 * np.sqrt(np.pi) * np.sqrt(sigma ** 2) ** 3)
    return [list(kernel)]


# We can create a 3D kernel using three one-dimensional kernels a, b, and c:
def create_three_dimensional_kernel(a, b, c):
    size = len(a[0])
    return np.outer(np.outer(a, b).ravel(), c).reshape(size, size, size)

# We can create a 3D gradient filter for the x, y, and t directions respectively
# using the seperability property of the Gaussian. Note that the dimensions of the
# matrix in this case are label (t, y, x).

three_dimensional_gradient_kernel_x = create_three_dimensional_kernel(
    one_dimensional_gaussian_filter(size, sigma),
    one_dimensional_gaussian_filter(size, sigma),
    one_dimensional_derivative_gaussian_filter(size, sigma))

three_dimensional_gradient_kernel_y = create_three_dimensional_kernel(
    one_dimensional_gaussian_filter(size, sigma),
    one_dimensional_derivative_gaussian_filter(size, sigma),
    one_dimensional_gaussian_filter(size, sigma))

three_dimensional_gradient_kernel_t = create_three_dimensional_kernel(
    one_dimensional_derivative_gaussian_filter(size, sigma),
    one_dimensional_gaussian_filter(size, sigma),
    one_dimensional_gaussian_filter(size, sigma))

V_x_gaussian = correlate(V, three_dimensional_gradient_kernel_x)
normalize_images(V_x_gaussian)

V_y_gaussian = correlate(V, three_dimensional_gradient_kernel_y)
normalize_images(V_y_gaussian)

V_t_gaussian = correlate(V, three_dimensional_gradient_kernel_t)
normalize_images(V_t_gaussian)

show_animation(V_x_gaussian, title="$V_x$ calculated with three dimensional fitler $G'(x)G(y)G(z)$.")
show_animation(V_y_gaussian, title="$V_y$ calculated with three dimensional fitler $G(x)G'(y)G(z)$.")
show_animation(V_t_gaussian, title="$V_t$ calculated with three dimensional fitler $G(x)G(y)G'(z)$.")

fig, axs = plt.subplots(2, 3, figsize=(30, 10))

def plot_image(X, ax, title):
    ax.imshow(X, cmap="gray")
    ax.set_title(title)
    ax.axis('off')

plot_image(V_x_gaussian[2], axs[0, 0], "$V_x$ calculated with the 3D Gaussian filter")
plot_image(V_x[2], axs[1, 0], "$V_x$ calculated manually")

plot_image(V_y_gaussian[2], axs[0, 1], "$V_y$ calculated with the 3D Gaussian filter")
plot_image(V_y[2], axs[1, 1], "$V_y$ calculated manually")

plot_image(V_t_gaussian[2], axs[0, 2], "$V_t$ calculated with the 3D Gaussian filter")
plot_image(V_t[2], axs[1, 2], "$V_t$ calculated manually")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
