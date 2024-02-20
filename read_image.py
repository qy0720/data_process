from PIL import Image
image_path = "/mnt/bn/qy-dcar-valume/data_process/20240218.jpeg"
img = Image.open(image_path)  # 打开图片
print(img.size)