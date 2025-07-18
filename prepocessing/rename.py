import os

def rename_mask_to_gtv_mask(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('_mask.npy') and not filename.endswith('_gtv_mask.npy'):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('_mask.npy', '_gtv_mask.npy')
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} -> {new_path}')

if __name__ == '__main__':
    target_dirs = [
        '/home/iujeong/brain_meningioma/slice/original/slice_test/npy',
        '/home/iujeong/brain_meningioma/slice/original/slice_test/png',
        '/home/iujeong/brain_meningioma/slice/original/slice_train/npy',
        '/home/iujeong/brain_meningioma/slice/original/slice_train/png',
    ]
    for d in target_dirs:
        rename_mask_to_gtv_mask(d)
