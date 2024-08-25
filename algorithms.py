import cv2
import numpy as np
from skimage.metrics import structural_similarity


def compare_p_hash(image1, image2):
    hash1 = p_hash(image1)
    hash2 = p_hash(image2)
    n3 = cmp_hash(hash1, hash2)
    return 1 - float(n3 / 64)


def round_clip(image, crop_width, crop_height):
    start_x = crop_width
    start_y = crop_height
    end_x = image.shape[1] - crop_width
    end_y = image.shape[0] - crop_height
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


# 均值哈希算法
def a_hash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值哈希算法
def d_hash(img):
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 感知哈希算法
def p_hash(img):
    img = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]
    res = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                res.append(1)
            else:
                res.append(0)
    return res


# 灰度直方图算法
def compare_hist_with_split(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                     (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def normalize_compared_hist_with_split(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += compare_hist_with_split(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# Hash值对比
# 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
# 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
# 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
def cmp_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def compare_ssim(image1, image2):
    before = image1
    after = image2
    before = before[0:after.shape[0], 0:after.shape[1]]
    after = after[0:before.shape[0], 0:before.shape[1]]
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(before_gray, after_gray, win_size=None, gradient=False, data_range=255,
                                          channel_axis=None, multichannel=False, gaussian_weights=False, full=True)
    return score


def is_similarity(image1, image2, debug=False):
    phash = compare_p_hash(image1, image2)
    ssim = compare_ssim(image1, image2)
    hist_with_split_rs = normalize_compared_hist_with_split(image1, image2)
    if debug:
        print("hist_with_split: %s, ssim: %s, phash: %s" % (hist_with_split_rs, ssim, phash))
    return hist_with_split_rs >= 0.8 or ssim > 0.7 or (hist_with_split_rs > 0.7 and phash > 0.89 and ssim > 0.2)


def compare_a_hash(image1, image2):
    hash1 = a_hash(image1)
    hash2 = a_hash(image2)
    n3 = cmp_hash(hash1, hash2)
    return 1 - float(n3 / 64)


def compare_d_hash(image1, image2):
    hash1 = d_hash(image1)
    hash2 = d_hash(image2)
    n3 = cmp_hash(hash1, hash2)
    return 1 - float(n3 / 64)


def runAllImageSimilaryFun(para1, para2, crop_width, crop_height):
    # 均值、差值、感知哈希算法三种算法值越小，则越相似,相同图片值为0
    # 三直方图算法和单通道的直方图 0-1之间，值越大，越相似。 相同图片为1

    # t1,t2   14;19;10;  0.70;0.75
    # t1,t3   39 33 18   0.58 0.49
    # s1,s2  7 23 11     0.83 0.86  挺相似的图片
    # c1,c2  11 29 17    0.30 0.31
    print("---------------------")
    img1 = round_clip(cv2.imread(para1), crop_width, crop_height)
    img2 = round_clip(cv2.imread(para2), crop_width, crop_height)
    print(is_similarity(img1, img2, True))

    n1 = compare_a_hash(img1, img2);
    print('均值哈希算法相似度aHash：', n1)

    n2 = compare_d_hash(img1, img2)
    print('差值哈希算法相似度dHash：', n2)

    n3 = compare_p_hash(img1, img2)
    print('感知哈希算法相似度pHash：', n3)

    n4 = normalize_compared_hist_with_split(img1, img2)
    print('三直方图算法相似度：', n4)

    n5 = compare_hist_with_split(img1, img2)
    print("单通道的直方图", n5)

    ssim1 = compare_ssim(img1, img2)
    print('ssim: ', ssim1)
    print("%d %d %d %.2f %.2f " % (n1, n2, n3, round(n4[0], 2), n5[0]))
    print("aHash: %.2f, dHash: %.2f, pHash: %.2f, 三直方图: %.2f, 单通道: %.2f, ssim: %.2f" %
          n1, 1 - float(n2 / 64), n3, round(n4[0], 2), n5[0], ssim1)


if __name__ == "__main__":
    runAllImageSimilaryFun("target/category/35.png", "target/category/36.png")
    runAllImageSimilaryFun("target/category/34.png", "target/grouped/34/107.png", 3, 3)
