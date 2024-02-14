# reference: https://github.com/MarcoForte/knn-matting
import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2

def get_features(image_RGB, mode='HSV'):
    # assume image is in RGB mode
    [h, w, c] = image_RGB.shape
    # get x, y coordinates and normalize
    x, y = np.unravel_index(np.arange(h * w), (h, w))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x = x / w
    y = y / h
    feature_vec = np.concatenate(([x, y]), axis=1)
    if mode == 'RGB':
        feature_vec = np.concatenate((feature_vec, image_RGB.reshape(-1, 3)), axis=1)
    elif mode == 'HSV':
        # convert RGB to HSV
        image_HSV = cv2.cvtColor(image_RGB.astype(np.float32), cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(image_HSV)
        H_cos = np.cos(H * 2 * np.pi).reshape(-1, 1)
        H_cos = (H_cos-(-1))/(1-(-1))
        H_sin = np.sin(H * 2 * np.pi).reshape(-1, 1)
        H_sin = (H_sin-(-1))/(1-(-1))
        S, V = S.reshape(-1, 1), V.reshape(-1, 1)
        feature_vec = np.concatenate((feature_vec, H_cos, H_sin, S, V), axis=1)        
    else:
        raise NotImplementedError
    return feature_vec

def my_get_knn(feature_vec, n_neighbors):
    idx = []
    for i in range(feature_vec.shape[0]):
        dist = np.sum((feature_vec - feature_vec[i]) ** 2, axis=1)
        idx.append(np.argsort(dist)[:n_neighbors])
    return np.array(idx)

def get_knn(feature_vec, obj_feature_vec, n_neighbors=10, n_jobs=8):
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(feature_vec)
    _, indices = knn.kneighbors(obj_feature_vec)
    return indices

def knn_matting(image_RGB, trimap, features_mode='HSV', n_neighbors=10, my_lambda=100):
    [h, w, c] = image_RGB.shape
    image_RGB, trimap = image_RGB / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)

    ####################################################
    # TODO: find KNN for the given image
    ####################################################
    X = get_features(image_RGB, mode=features_mode)
    
    n_jobs = 8
    indices = get_knn(X, X, n_neighbors=n_neighbors, n_jobs=n_jobs)
    # indices = my_get_knn(X, n_neighbors=n_neighbors)
    
    ####################################################
    # TODO: compute the affinity matrix A
    #       and all other stuff needed
    ####################################################
    i = np.repeat(np.arange(h * w), n_neighbors)
    j = indices.reshape(-1)
    k = 1-np.linalg.norm(X[i] - X[j], axis=1)/X.shape[1]
    A = scipy.sparse.coo_matrix((k, (i, j)), shape=(h * w, h * w))

    D_non = scipy.sparse.diags(A.sum(axis=1).flatten().tolist()[0])
    L = D_non - A

    # (L+λD)α = λv
    # m is a binary vector of indices of all the marked-up pixels
    m = (foreground+background).flatten()
    # where D = diag(m)
    D = scipy.sparse.diags(m)

    # where v is a binary vector of pixel indices corresponding to user markups for a given layer
    v = foreground.flatten()
    # Note that H = 2(L+λD) and c = 2λv
    H = 2 * (L + my_lambda * D)
    c = 2 * (my_lambda * np.transpose(v))

    # optimal solution is H^{-1}c, solve Hx = c  

    ####################################################
    # TODO: solve for the linear system,
    #       note that you may encounter en error
    #       if no exact solution exists
    ####################################################

    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = scipy.sparse.linalg.spsolve(H, c)
        pass
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = x[0]
        pass
    alpha = np.clip(alpha, 0, 1)
    alpha = alpha.reshape(h, w)
    return alpha


if __name__ == '__main__':
    image_name = 'multiple_troll'
    bg_image_name = 'table'
    features = 'RGB'
    n_neighbors = 5
    
    image = cv2.imread('./image/{}.png'.format(image_name))
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    trimap = cv2.imread('./trimap/{}.png'.format(image_name), cv2.IMREAD_GRAYSCALE)

    alpha = knn_matting(image_RGB, trimap, features_mode=features, n_neighbors=n_neighbors)
    alpha = alpha[:, :, np.newaxis]
    alpha3 = np.repeat(alpha, 3, axis=2)

    ####################################################
    # TODO: pick up your own background image, 
    #       and merge it with the foreground
    ####################################################
    bg_image = cv2.imread('./background/{}.png'.format(bg_image_name))
    bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    bg_image_RGB = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

    # I = αF + (1 − α)B
    result = []
    result = alpha3 * image_RGB + (1 - alpha3) * bg_image_RGB
    result = result.astype(np.uint8)
    result_BGR = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./result/{}_{}_k{}_{}.png'.format(image_name, bg_image_name, n_neighbors, features), result_BGR)
    cv2.imwrite('./result/{}_{}_k{}_{}_alpha.png'.format(image_name, bg_image_name, n_neighbors, features), alpha * 255)