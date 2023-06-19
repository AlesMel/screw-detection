import matplotlib.pyplot as plt
import albumentations as A
import cv2


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


transform = A.Compose(
    [
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75
        ),
        A.Blur(blur_limit=3),
        A.HueSaturationValue(),
    ]
)

#

for i in range(1, 10):
    path = "sample_img/type_00" + str(i)
    base_img = path + f"/{i}_1.jpeg"
    for j in range(2, 19):
        image = cv2.imread(base_img)
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = transform(image=image)["image"]
        cv2.imwrite(path + f"/{i}_{j}.jpeg", augmented_image)
