import cv2
from scipy import ndimage
import numpy as np

def get_normal_map(img):
    img = img.astype(np.float)
    img = img / 255.0
    img = - img + 1
    img[img < 0] = 0
    img[img > 1] = 1
    return img

def get_gray_map(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    highPass = gray.astype(np.float)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    highPass = highPass[None]
    return highPass.transpose((1,2,0))

def get_light_map(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float)
    highPass = highPass / 128.0
    highPass = highPass[None]
    return highPass.transpose((1,2,0))

def get_light_map_single(img):
    gray = img
    gray = gray[None]
    gray = gray.transpose((1,2,0))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = gray.reshape((gray.shape[0],gray.shape[1]))
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float)
    highPass = highPass / 128.0
    return highPass

def get_light_map_drawer(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    highPass = gray.astype(int) - blur.astype(int) + 255
    highPass[highPass < 0 ] = 0
    highPass[highPass > 255] = 255
    highPass = highPass.astype(np.float)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    highPass = highPass[None]
    return highPass.transpose((1,2,0))

def get_light_map_drawer2(img):
    ret = img.copy()
    ret=ret.astype(np.float)
    ret[:, :, 0] = get_light_map_drawer3(img[:, :, 0])
    ret[:, :, 1] = get_light_map_drawer3(img[:, :, 1])
    ret[:, :, 2] = get_light_map_drawer3(img[:, :, 2])
    ret = np.amax(ret, 2)
    return ret

def get_light_map_drawer3(img):
    gray = img
    blur = cv2.blur(gray,ksize=(5,5))
    highPass = gray.astype(int) - blur.astype(int) + 255
    highPass[highPass < 0 ] = 0
    highPass[highPass > 255] = 255
    highPass = highPass.astype(np.float)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    return highPass

def normalize_pic(img):
    img = img / np.max(img)
    return img

def superlize_pic(img):
    img = img * 2.33333
    img[img > 1] = 1
    return img

def mask_pic(img,mask):
    mask_mat = mask
    mask_mat = mask_mat.astype(np.float)
    mask_mat = cv2.GaussianBlur(mask_mat, (0, 0), 1)
    mask_mat = mask_mat / np.max(mask_mat)
    mask_mat = mask_mat * 255
    mask_mat[mask_mat<255] = 0
    mask_mat = mask_mat.astype(np.uint8)
    mask_mat = cv2.GaussianBlur(mask_mat, (0, 0), 3)
    mask_mat = get_gray_map(mask_mat)
    mask_mat = normalize_pic(mask_mat)
    mask_mat = resize_img_512(mask_mat)
    super_from = np.multiply(img, mask_mat)
    return super_from

def resize_img_512(img):
    zeros = np.zeros((512,512,img.shape[2]), dtype=np.float)
    zeros[:img.shape[0], :img.shape[1]] = img
    return zeros

def resize_img_512_3d(img):
    zeros = np.zeros((1,3,512,512), dtype=np.float)
    zeros[0 , 0 : img.shape[0] , 0 : img.shape[1] , 0 : img.shape[2]] = img
    return zeros.transpose((1,2,3,0))

def show_active_img_and_save(name,img,path):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    cv2.imshow(name,mat)
    cv2.imwrite(path,mat)
    return

def denoise_mat(img,i):
    return ndimage.median_filter(img, i)

def show_active_img_and_save_denoise(name,img,path):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name,mat)
    cv2.imwrite(path,mat)
    return

def show_active_img_and_save_denoise_filter(name,img,path):
    mat = img.astype(np.float)
    mat[mat<0.18] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name,mat)
    cv2.imwrite(path,mat)
    return

def show_active_img_and_save_denoise_filter2(name,img,path):
    mat = img.astype(np.float)
    mat[mat<0.1] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name,mat)
    cv2.imwrite(path,mat)
    return

def show_active_img(name,img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    cv2.imshow(name,mat)
    return

def get_active_img(img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat

def get_active_img_fil(img):
    mat = img.astype(np.float)
    mat[mat < 0.18] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat

def show_double_active_img(name,img):
    mat = img.astype(np.float)
    mat = mat * 128.0
    mat = mat + 127.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    cv2.imshow(name,mat.astype(np.uint8))
    return

def debug_pic_helper():
    for index in range(1130):
        gray_path = 'data\\gray\\'+str(index)+'.jpg'
        color_path = 'data\\color\\' + str(index) + '.jpg'

        mat_color = cv2.imread(color_path)
        mat_color=get_light_map(mat_color)
        mat_color=normalize_pic(mat_color)
        mat_color=resize_img_512(mat_color)
        show_double_active_img('mat_color',mat_color)

        mat_gray = cv2.imread(gray_path)
        mat_gray=get_gray_map(mat_gray)
        mat_gray=normalize_pic(mat_gray)
        mat_gray = resize_img_512(mat_gray)
        show_active_img('mat_gray',mat_gray)

        cv2.waitKey(1000)

def preprocess(from_mat):
    width = float(from_mat.shape[1])
    height = float(from_mat.shape[0])
    new_width = 0
    new_height = 0
    if (width > height):
        from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512

def recover(mat,threshold=0.07):
    mat = mat.astype(np.float)
    mat[mat<threshold] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    out =cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
    return out

import matplotlib.pyplot as plt

def gray_print(img):
    plt.imshow(img,'gray')
def rgb_print(img):
    img_rgb=img[:,:,[2,1,0]]
    plt.imshow(img_rgb,'gray')

import os
def under_file(file_dir):
    '''
    get all the image abs path under a file
    '''
    if not os.path.isfile(file_dir):
        filepaths=[]
        for filename in os.listdir(file_dir):
            if "csv" not in filename:
                filepath=os.path.join(file_dir,filename)
                filepaths.extend(under_file(filepath))
    else:
        filepaths=[file_dir]
    return filepaths


def grab_pic(func,paths,array):
    rowm,colm=array.shape
    thiscol=[]
    for i in range(rowm):
        thisline=[]
        for j in range(colm):
            pic= func(paths,array[i][j],0.01)
            thisline.append(pic)
        thiscol.append(np.concatenate(thisline,axis=1))
    return np.concatenate(thiscol)

def get_edge(image,para=(7,7)):
    pic_preprocessed  = cv2.cvtColor(cv2.GaussianBlur(image, para, 0), cv2.COLOR_BGR2GRAY)
    pic_edges = cv2.bitwise_not(cv2.Canny(pic_preprocessed, threshold1=20, threshold2=60))
    pic_out=cv2.cvtColor(pic_edges, cv2.COLOR_GRAY2RGB)
    return pic_out

def get_edge2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(img, ret * 0.5, ret)
    out=cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return out

def prinable(sketch,h,w):
    out =cv2.resize(sketch,(h,w))
    if len(sketch.shape)==2:out=cv2.merge([sketch,sketch,sketch])
    return out.astype('int')

def highPass(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 10)
    highPass = gray.astype(int) - blur.astype(int) + 255
    highPass[highPass < 0 ] = 0
    highPass[highPass > 255] = 255
    highPass = highPass.astype(np.float)
    return highPass

def test_look(imagepaths,num,threshold=0.07):
    image=cv2.imread(imagepaths[num])
    w,h,_=image.shape
    sketch  =get_sketch(image,threshold);
    sketch =prinable(sketch,h,w)
    row1=np.concatenate([image,sketch],axis=1)
    return row1

def train_look(pairpaths,num,threshold=0.07):
    pair = pairpaths[num]
    image= cv2.imread(pair[0])
    w,h,_ =image.shape
    sketch  =get_sketch(image,threshold);
    sketch  =prinable(sketch,h,w)
    true_sketch=cv2.imread(pair[1])
    true_sketch=highPass(true_sketch)
    true_sketch=prinable(true_sketch,h,w)
    row1=np.concatenate([image,sketch,true_sketch],axis=1)
    return row1
