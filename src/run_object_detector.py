from utils.image_processing import *

# images_path = '/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/test/'
# renamed_images_path = '/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/test_renamed/'

# for bclass in classes:


#   for index, each_test_scene in enumerate(os.listdir(renamed_images_path + bclass)):
#     os.rename(renamed_images_path + bclass + '/' + each_test_scene, renamed_images_path + bclass + '/' + 't_' + bclass[0] + '_' + str(index) +'.jpg')

# image_path = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/to be detected"

# for every_file in os.listdir(image_path):
#     resize_and_save(image_path + '/' + every_file, image_path + '/' + every_file, resized_street_image_dims_width, resized_street_image_dims_height)


resized_street_image_dims_width = 700
resized_street_image_dims_height = 450
classes = ["residential", "mixed", "commercial","others","industrial"]
# images_path = '/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/test_renamed/all'
# resized_images_path = '/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/test_renamed/all_resized'

# for every_file in os.listdir(images_path):
#     resize_and_save(images_path + '/' + every_file, resized_images_path + '/' + every_file, resized_street_image_dims_width, resized_street_image_dims_height)

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

resized_img_size = 500
cropped_dims = 800
frequency_dict = dict()
predicted_dict = {}
resized_street_image_dims = 800
instances_folder = "/content/gdrive/Othercomputers/My Laptop/Dataset Images/instances/test"

resized_images_path = '/content/gdrive/Othercomputers/My Laptop/Dataset Images/scenes/test_renamed/all_resized'

for bclass in classes:

  for i in os.listdir(resized_images_path +'/'+ bclass + '/'):
    
    predicted_dict[bclass +'/' + i] = []    
    bounding_objects = []
    
    # for each image, run the detector and get the bounding boxes and corresponding
    img = Image.open(resized_images_path + '/'+ bclass + '/' + i)
    bounding_boxes, raw_bounding_classes, class_scores = run_detector(detector, resized_images_path + '/' + bclass + '/' + i)
    
    # for each of the objects detected, check if house, skyscraper or building, crop according to priority, and save them to the cropped_folder
    for index, i3 in enumerate(raw_bounding_classes):
      i1 = str(i3)[2:-1].lower()
      if i1 in ['house', 'building', 'skyscraper', 'tower'] and class_scores[index] > 0.3:
        final_path = instances_folder + "/" + bclass + '/' + str(index) + '_' + i
        expanded_boxes = np.asarray(tuple(expansions)) * np.asarray(bounding_boxes[index])
        house_bound = crop_bounding_box(img, expanded_boxes)
        house_bound = fit_and_save(house_bound, final_path, 500, 700)
    
    for objects in range(len(raw_bounding_classes)):
      if class_scores[objects] >= 0.4:
        bounding_objects.append(str(raw_bounding_classes[objects])[2:-1].lower())
    
    # record the number of objects detected
    frequency_dict[bclass +'/' + i] = Counter(bounding_objects)

expansions = [resized_street_image_dims_height, resized_street_image_dims_width, resized_street_image_dims_height, resized_street_image_dims_width]

