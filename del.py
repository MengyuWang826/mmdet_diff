import os 


if __name__ == '__main__':
    all_files = os.listdir('/home/wmy/data/mmdet_merge/mmdet_diff')
    for filename in all_files:
        if 'jpg' in filename or 'png' in filename:
            os.remove(filename)
    a = 1