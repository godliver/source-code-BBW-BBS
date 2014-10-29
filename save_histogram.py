from utils_shape_and_color import save_histogram
import glob
healthy = './healthy/'
bbw = './bbw/'
bbs = './bbs/'
healthy_files = glob.glob(healthy + '*.PNG')
bbw_files = glob.glob(bbw + '*.PNG')
bbs_files = glob.glob(bbs + '*.PNG')

save_histogram(healthy_files,bbw_files,bbs_files)
number_of_images = len(healthy_files) + len(bbw_files) + len(bbs_files)
