import os
from config import params

com_path = params['compare_path'] + params['dataset'] + '/'
if not os.path.exists(os.path.join(com_path)):
    os.makedirs(com_path)