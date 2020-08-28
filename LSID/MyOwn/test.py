# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
import Helpers.ModelHelper as sid
from os.path import basename

input_image_path = 'TestImages/night_owls_test.png'
checkpoint_dir = '../checkpoint/Sony/'
result_dir = 'TestResults/'

input_image_basename = basename(input_image_path)
input_image_basename = input_image_basename[:-4]
save_path = result_dir + input_image_basename + "_result.png"

sid_model = sid.SIDModel(checkpoint_dir)

sid_model.run_png(input_image_path,save_path)
