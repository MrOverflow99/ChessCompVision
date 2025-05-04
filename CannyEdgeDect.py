import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    size = int(size)
    x, y = np.mgrid[-size//2+1:size//2+1, -size//2+1:size//2+1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()

def convolve(image, kernel):
    padding = kernel.shape[0] // 2
    padded_img = np.zeros((image.shape[0] + 2*padding, image.shape[1] + 2*padding))
    padded_img[padding:-padding, padding:-padding] = image
    result = np.zeros_like(image, dtype=np.float32)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            result[x, y] = (kernel * padded_img[x:x+2*padding+1, y:y+2*padding+1]).sum()
    return result

def sobel_filters(image):
    Kx = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]], np.float32)
    
    Ky = np.array([[ -1, -2, -1],
                   [  0,  0,  0],
                   [  1,  2,  1]], np.float32)
    
    Ix = convolve(image, Kx)
    Iy = convolve(image, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    theta = np.rad2deg(theta) % 180
    return (G, theta)

def non_max_suppression(G, theta):
    nms = np.zeros_like(G)
    angle = theta.copy()
    
    angle[angle < 22.5] = 0
    angle[(angle >= 22.5) & (angle < 67.5)] = 45
    angle[(angle >= 67.5) & (angle < 112.5)] = 90
    angle[(angle >= 112.5) & (angle < 157.5)] = 135
    
    for i in range(1, G.shape[0]-1):
        for j in range(1, G.shape[1]-1):
            if angle[i,j] == 0:
                if G[i,j] >= G[i, j+1] and G[i,j] >= G[i, j-1]:
                    nms[i,j] = G[i,j]
            elif angle[i,j] == 45:
                if G[i,j] >= G[i-1, j+1] and G[i,j] >= G[i+1, j-1]:
                    nms[i,j] = G[i,j]
            elif angle[i,j] == 90:
                if G[i,j] >= G[i+1, j] and G[i,j] >= G[i-1, j]:
                    nms[i,j] = G[i,j]
            elif angle[i,j] == 135:
                if G[i,j] >= G[i-1, j-1] and G[i,j] >= G[i+1, j+1]:
                    nms[i,j] = G[i,j]
    return nms

def threshold(image, low, high):
    res = np.zeros_like(image)
    weak = 50
    strong = 255
    
    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image < high) & (image >= low))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return (res, strong)

def hysteresis(image, weak=50):
    res = image.copy()
    strong = 255
    weak_i, weak_j = np.where(res == weak)
    queue = []
    for i, j in zip(weak_i, weak_j):
        if (res[i,j-1] == strong or 
            res[i,j+1] == strong or 
            res[i-1,j] == strong or 
            res[i+1,j] == strong or 
            res[i-1,j-1] == strong or 
            res[i-1,j+1] == strong or 
            res[i+1,j-1] == strong or 
            res[i+1,j+1] == strong):
            queue.append((i,j))
    
    while queue:
        i,j = queue.pop(0)
        res[i,j] = strong
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (0 <= i+dx < res.shape[0] and 
                    0 <= j+dy < res.shape[1] and 
                    res[i+dx, j+dy] == weak):
                    res[i+dx, j+dy] = strong
                    queue.append((i+dx, j+dy))
    
    res[res == weak] = 0
    return res

def canny(image, sigma=1, kernel_size=5, low_threshold=0.1, high_threshold=0.2):
    blurred = convolve(image, gaussian_kernel(kernel_size, sigma))
    G, theta = sobel_filters(blurred)
    nms = non_max_suppression(G, theta)
    tresh, strong = threshold(nms, low_threshold*nms.max(), high_threshold*nms.max())
    final = hysteresis(tresh)
    return final

def find_contours(edge_image):
    contours = []
    visited = np.zeros_like(edge_image, dtype=bool)
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),         (0,1),
                  (1,-1),  (1,0), (1,1)]
    
    for i in range(1, edge_image.shape[0]-1):
        for j in range(1, edge_image.shape[1]-1):
            if edge_image[i,j] == 255 and not visited[i,j]:
                contour = []
                current = (i,j)
                prev_dir = 0
                contour.append(current)
                visited[current[0], current[1]] = True
                while True:
                    found = False
                    for d in range(8):
                        dx, dy = directions[(prev_dir - d) %8]
                        ni, nj = current[0]+dx, current[1]+dy
                        if (0 <= ni < edge_image.shape[0] and 
                            0 <= nj < edge_image.shape[1] and 
                            edge_image[ni,nj] == 255 and 
                            not visited[ni,nj]):
                            current = (ni, nj)
                            visited[ni,nj] = True
                            contour.append(current)
                            prev_dir = (prev_dir - d) %8
                            found = True
                            break
                    if not found:
                        break
                if len(contour) > 10:
                    contours.append(contour)
    return contours

# Example usage
if __name__ == "__main__":
    # Load an image (replace with your own grayscale image)
    # This example uses a simple test pattern
    from skimage.data import camera
    image = camera()  # Grayscale test image
    
    # Perform Canny Edge Detection
    edges = canny(image, sigma=1.4, kernel_size=5, low_threshold=0.05, high_threshold=0.15)
    
    # Find contours
    contours = find_contours(edges)
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection Result')
    
    # Plot contours
    for cnt in contours[:5]:  # Plot first 5 contours for demonstration
        cnt = np.array(cnt)
        plt.plot(cnt[:,1], cnt[:,0], 'r', linewidth=0.5)
    
    plt.show()