import urllib.request, os, zipfile

URLS = {
    'https://github.com/hmarichal93/apd/releases/download/v1.0_icpr_2024_submission/all.zip': 'dataset/all.zip',
}

for url, destination_zipfile in URLS.items():
    destination_dir = os.path.dirname(destination_zipfile)
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as f:
        os.makedirs(destination_dir, exist_ok=True)
        open(destination_zipfile, 'wb').write(f.read())

    print(f'Extracting into {os.path.abspath(destination_dir)}')
    zipfile.ZipFile(destination_zipfile).extractall(destination_dir)
    os.remove(destination_zipfile)

print('Done')


