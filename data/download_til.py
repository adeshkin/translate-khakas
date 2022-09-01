import os


def main():
    base_url = 'gs://til-corpus/corpus'
    
    save_dir = 'til_data'
    source = 'tr'
    target = 'ru'

    for split in ['train', 'dev', 'test']:
        save_path = f'{save_dir}/{source}-{target}/{split}'
        os.makedirs(save_path)
        download_path = f'{base_url}/{split}/{source}-{target}'
        print(f'Downloading {split} files...')

        # gsutil works in google colaboratory
        os.system(f'gsutil -m cp -r {download_path} {save_path}')
        
        
if __name__ == '__main__':
    main()
