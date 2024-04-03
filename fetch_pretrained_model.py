import urllib.request, os

URLS = {
    'https://github.com/hmarichal93/apd/releases/download/v1.0_icpr_2024_submission/all_best_yolov8.pt' : 'checkpoints/yolo/all_best_yolov8.pt',
}

for url, destination in URLS.items():
    print(f'Downloading {url} ...')
    with urllib.request.urlopen(url) as f:
        os.makedirs( os.path.dirname(destination), exist_ok=True )
        open(destination, 'wb').write(f.read())
