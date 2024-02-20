# 读取图片  
src_image = cv2.imread(image_path)
src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)  

dst_image = cv2.imread(os.path.join(output_path, basename))
dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)  

src_mask = cv2.imread(mat_path)  # mask_image应为灰度图像  

# 确保mask是单通道的  
if src_mask.ndim == 3:  
    src_mask = cv2.cvtColor(src_mask, cv2.COLOR_BGR2GRAY)  

# 创建一个和原始图片相同大小的掩码，其中mask区域的像素值为255，其余为0  
roi_mask = np.zeros_like(src_mask)  
roi_mask[src_mask > 0] = 255

# 将替换区域应用到原始图片  
dst_image = dst_image.copy()  
dst_image[roi_mask > 0] = src_image[roi_mask > 0]


image = Image.fromarray(dst_image.astype(np.uint8))
image.save(os.path.join(output_path, basename))
