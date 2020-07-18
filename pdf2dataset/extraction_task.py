import os
import traceback

import numpy as np
import pytesseract
import cv2
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
import base64


class ExtractionTask:

    def __init__(self, doc, page, lang='por', img_column='no'):
        self.doc = doc
        self.page = page
        self.lang = lang
        self.img_column = img_column

    def preprocess_image(self, img):
        tsh = np.array(img.convert('L'))
        tsh = cv2.adaptiveThreshold(
            tsh, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            97, 50
        )

        erd = cv2.erode(
            tsh,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
            iterations=1
        )

        return erd

    def ocr_image(self, img):
        # So pytesseract uses only one core per worker
        os.environ['OMP_THREAD_LIMIT'] = '1'
        return pytesseract.image_to_string(img, lang=self.lang)

    def encode_image(self, img):
        img_resized = cv2.resize(img, (224,224))
        img_encoded = cv2.imencode('.jpg', img_resized)[1].tostring()
        img_as_b64 = base64.b64encode(img_encoded)
        return img_as_b64
        
    def get_page_img(self):
        img = convert_from_path(
            self.doc,
            first_page=self.page,
            single_file=True,
            size=(None, 1100),
            fmt='jpeg'
        )

        return img[0]

    def process(self):
        text, img_encoded, img_preprocessed, error = None, None, None, None

        # Ray can handle the exceptions, but this makes switching to
        #   multiprocessing easy
        try:
            img = self.get_page_img()

            # TODO: Use OCR?

            img_preprocessed = self.preprocess_image(img)
            text = self.ocr_image(img_preprocessed)
            if self.img_column == 'yes':
                img_encoded = self.encode_image(img_preprocessed)

        except (PDFPageCountError, PDFSyntaxError):
            error = traceback.format_exc()

        if self.img_column == 'yes':
            return (text, img_encoded), error
        else:
            return text, error
