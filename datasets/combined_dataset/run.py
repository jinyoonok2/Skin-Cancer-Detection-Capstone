# run_all.py

import runpy

# Running the scripts in order
runpy.run_path('default_remove_roboflow_name.py')
runpy.run_path('default_class_correction.py')
runpy.run_path('default_group_split.py')
runpy.run_path('train_magnification.py')
runpy.run_path('check_empty.py')