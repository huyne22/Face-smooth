# Save image adapted from https://stackoverflow.com/a/56680778/10796680

# System imports
import os

# Third-party imports
import cv2
import numpy as np

from detector import detect, smooth

#thay đổi ảnh theo chiều rộng chiều cao cho trước
def resize_image(image, width=None, height=None):
    """
    Resize image with proportionate scaling. e.g. If 
    only width is given, height will automatically 
    proportionally scale.

    Source
    ------
    https://stackoverflow.com/a/56859311/10796680

    Parameters
    ----------
    img : np.array [H, W, 3]
        RGB image

    Returns
    -------
    image shape : int, int
        height and width of image
    """
    dim = None
    (h, w) = image.shape[:2]
    # Kiểm tra xem có đặt chiều rộng (width) và chiều cao (height) hay không
    if width is None and height is None:
         # Nếu không có giá trị width và height được đặt, trả về hình ảnh gốc
        return image
    # Nếu chỉ có một trong hai giá trị width hoặc height được đặt
    if width is None:
        # Tính tỉ lệ giữa chiều cao mới và chiều cao gốc
        r = height / float(h)
        # Tính kích thước mới dựa trên tỉ lệ và chiều rộng gốc
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        # Thực hiện thay đổi kích thước sử dụng phương pháp nội suy INTER_AREA
    resized = cv2.resize(image, 
                         dim, 
                         interpolation=cv2.INTER_AREA)
    # Trả về hình ảnh đã được thay đổi kích thước
    return resized


def check_img_size(img):
    """
    Verifies that the image is 360x540 or smaller
    to help the detector find faces.
    """
    # Lấy kích thước hình ảnh
    height, width = img.shape[:2]

    # Kiểm tra nếu chiều cao (height) hoặc chiều rộng (width) vượt quá giới hạn
    if height > 720 or width > 1080:
        # Nếu vượt quá giới hạn, thực hiện thay đổi kích thước sử dụng hàm resize_image
        img = resize_image(img, 
                           width=720 if width > 720 else None, 
                           height=1080 if height > 1080 else None)

    # Trả về hình ảnh, có thể đã được thay đổi kích thước
    return img


def process_image(input_img, cfg, net):
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    output_img : np.array [H,W,3]
        BGR image of face

    cfg : dict
        Dictionary of configurations

    bboxes : list [[x1, y1, x2, y2],...]
        List of lists of bbox coordinates

    Returns
    -------
    images : tuple
        Tuple of BGR images
    """
   # Đảm bảo rằng kích thước của hình ảnh nhỏ hơn 1081px chiều rộng
    input_img = check_img_size(input_img)
     # Phát hiện khuôn mặt và lấy bounding boxes
    detected_img, bboxes = detect.detect_face(cfg, net, input_img)
    # Làm mượt khuôn mặt và trả về các bước xử lý
    output_img, roi_img, hsv_mask, smoothed_roi = smooth.smooth_face(cfg,
                                                                     input_img, 
                                                                     bboxes)
    # Vẽ bounding boxes lên hình ảnh đầu ra
    output_w_bboxes = draw_bboxes(output_img, cfg, bboxes)
    # Trả về tuple chứa các hình ảnh kết quả từ quá trình xử lý
    return (input_img, detected_img, roi_img, hsv_mask, 
            smoothed_roi, output_w_bboxes, output_img)


def load_image(path):
    """
    Read an image using OpenCV

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    image : np.array [H,W,3]
        RGB image
    """
     # Sử dụng OpenCV để đọc hình ảnh từ đường dẫn
    return cv2.imread(path)

#ok có file1 -> add file 2
def create_img_output_path(filename):
    """
    Checks if filename already exists and appends int to
    end of path if path already exists.

    Parameters
    ----------
    filename : str
        Path to file.

    Returns
    -------
    filename : str
        Path to file which is confirmed to not exist yet.
    """
    counter = 0
    filename = filename + '{}.jpg'
      # Nếu một tệp có tên tương tự đã tồn tại, tăng biến đếm lên 1 
    while os.path.isfile(filename.format(counter)):
        counter += 1
    # Áp dụng biến đếm vào filename
    filename = filename.format(counter)
    return filename.format(counter)


def save_image(filename, img):
    """
    Save an image using OpenCV

    Parameters
    ----------
    output_dir : str
        Name to save image as
    filename : str
        Name to save image as
    img : str
        Name to save image as

    Returns
    -------
    Bool : bool
        True if image save was success
    """
     # Tạo tên tệp mới để đảm bảo không trùng lặp
    filename = create_img_output_path(filename)
    # Save image
    return cv2.imwrite(filename, img)


def get_height_and_width(img):
    """
    Retrieve height and width of image

    Parameters
    ----------
    img : np.array [H, W, 3]
        RGB image

    Returns
    -------
    image shape : int, int
        height and width of image
    """
    return img.shape[0], img.shape[1]


def concat_imgs(imgs):
    """
    Concatenates tuple of images.

    Parameters
    ----------
    imgs : tuple
        tuple of BGR images

    Returns
    -------
    combined_img : BGR image
        Image of horizontally stacked images
    """
    # Nối các hình ảnh theo chiều ngang
    return np.concatenate(imgs, axis=1)

def save_steps(filename, all_img_steps, output_height):
    """
    Resizes and concatenates tuple of images.

    Parameters
    ----------
    filename : str
        Output filename

    all_img_steps : tuple
        Tuple of BGR images

    output_height : int
        Height of output image

    Returns
    -------
    img_saved : bool
        True if successful save
    """
    # Thay đổi kích thước ảnh trong tuple
    resized_imgs = tuple(resize_image(img, None, output_height) 
                                      for img in all_img_steps)
     # Nối các ảnh theo chiều ngang
    combined_imgs = concat_imgs(resized_imgs)
   # Lưu ảnh đã nối
    return save_image(filename, combined_imgs)

#vẽ hộp giới hạn 
def draw_bboxes(output_img, cfg, bboxes):
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    output_img : np.array [H,W,3]
        BGR image of face

    cfg : dict
        Dictionary of configurations

    bboxes : list [[x1, y1, x2, y2],...]
        List of lists of bbox coordinates

    Returns
    -------
    image : np.array [H,W,3]
        BGR image with bounding boxes
    """
    # Tạo bản sao của ảnh
    output_w_bboxes = output_img.copy()
     # Lấy chiều cao và chiều rộng của ảnh
    img_height, img_width = get_height_and_width(output_w_bboxes)
     # Vẽ các hộp giới hạn
    for i in range(len(bboxes)):
        top_left = (bboxes[i][0], bboxes[i][1])
        btm_right = (bboxes[i][2], bboxes[i][3])
        cv2.rectangle(output_w_bboxes, 
                      top_left, 
                      btm_right, 
                      cfg['image']['bbox_color'], # Màu của hộp giới hạn
                      2) # Độ dày đường vẽ
    return output_w_bboxes        

def check_if_adding_bboxes(args, img_steps):
    """
    Check if --show-detections flag is given. 
    If it is, return the image with bboxes.

    Parameters
    ----------
    args : Namespace object
        ArgumentParser

    img_steps : tuple
        Tuple of image steps

    Returns
    -------
    configs : dict
        A dictionary containing the configs
    """
    # Nếu cờ --show-detections được kích hoạt, trả về ảnh có chứa các hộp giới hạn
    if args.show_detections:
        return img_steps[5]
    else:
        # Nếu không, trả về ảnh cuối cùng sau quá trình xử lý
        return img_steps[6]


