
# G1 = one_dimensional_gaussian_filter(size, sigma)
#
#
# V_gaussian1 = filter_images(V, G1, axis=0)
# V_gaussian1 = filter_images(V_gaussian1, np.transpose(G1), axis=0)
# show_animation(V_gaussian1, title=f"1D Gaussian filter of length {size} applied twice")
#
# G2 = two_dimensional_gaussian_filter(size, sigma)
# V_gaussian2 = filter_images(V, G2, axis=0)
# show_animation(V_gaussian2, title=f"2D Gaussian filter of dimension {size}X{size}")
#
# # Difference of the two methods (black indicates no difference)
# show_animation(np.abs(V_gaussian1-V_gaussian2), title="Difference between applying 1D Gauss twice and 2D Gauss once")
#
# # 3D Gaussian filter:
#
# G = one_dimensional_gaussian_filter(size, sigma)
#
# V_gaussian = filter_images(V, G, axis=0)
# V_gaussian = filter_images(V_gaussian, np.transpose(G), axis=0)
# V_gaussian = filter_images(V_gaussian, G, axis=2)
#
# show_animation(V_gaussian)


def create_three_dimensional_kernel(a, b, c):
    size = len(a[0])
    return np.outer(np.outer(a, b).ravel(), c).reshape(size, size, size)


def create_two_dimensional_kernel(a, b):
    size = len(a[0])
    return np.outer(a, b)


kernel = create_three_dimensional_kernel(one_dimensional_gaussian_filter(4, 1),
                                         one_dimensional_gaussian_filter(4, 1),
                                         one_dimensional_derivative_gaussian_filter(4, 1))

# plt.plot(g[0], marker=".")
# plt.show()
V_g = correlate(V, kernel)
for t in range(V_g.shape[0]):
    min_val, max_val = np.min(V_g[t, :, :]), np.max(V_g[t, :, :])
    if max_val > min_val:
        V_g[t, :, :] = (V_g[t, :, :] - min_val) / (max_val - min_val)
    else:
        V_g[t, :, :] = np.zeros_like(V_g[t, :, :])

# V_g = filter_images(V, g, axis=0)
show_animation(V_g)

exit()

gaussian_kernel = one_dimensional_gaussian_filter(6, 1)
# First apply 1D filter in x (axis 2) direction then in y (axis 1) direction
V_g1 = filter_images(filter_images(V, gaussian_kernel, axis=2), gaussian_kernel, axis=1)

gaussian_kernel = two_dimensional_gaussian_filter(6, 1)
V_g2 = filter_images(V, gaussian_kernel, axis=0)

show_animation(abs(V_g1 - V_g2))

filter = one_dimensional_derivative_gaussian_filter(60, 1)
V_x = filter_images(V, filter, axis=0)

plt.hist(filter, bins=60)
plt.show()

show_animation(V_x)