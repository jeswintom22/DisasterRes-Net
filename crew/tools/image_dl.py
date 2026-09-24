from crew.vision.fetch import safe_fetch_image

class ImageDownloaderTool:
    name = "safe_image_download"
    def run(self, url: str):
        return safe_fetch_image(url)
