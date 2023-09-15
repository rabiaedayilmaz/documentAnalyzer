from PyPDF2 import PdfReader
import re
import pytesseract
from PIL import Image
import cv2
from io import BytesIO
import numpy as np
import pandas as pd

def pdf2txt(pdf_path: str) -> list:
    """
        Parameters:
            pdf_path: a str of path for pdf file
        Return:
            file_text: a list of str for pages of pdf file
    """
    reader = PdfReader(pdf_path)

    file_text = []

    n_pages = len(reader.pages)

    for page_num in range(n_pages):
        page = reader.pages[page_num]
        page_text = page.extract_text()
        file_text.append(page_text)
    return file_text


'''def img2txt(png_path: str):
    """
        Reads image. Converts it to gray scale. 
        Apply gaussian blur, followed by Otsu's threshold to obtain binary image.
        Apply morphological operations to remove noise and artifacts.

        Supported file types: PNG, JPEG, TIFF, JPEG 2000, GIF, WebP, BMP, PNM

        How to Build `tesseract`:
            brew install wget
            brew install tesseract
            brew install tesseract-lang
            wget -O /usr/local/codebase/documentSummarizer/tur.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/tur.traineddata

        Parameters:
            png_path: a str of path for png file
        Return:
            text: a str of text on the image
    """
    image = cv2.imread(png_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    text = pytesseract.image_to_string(invert, lang="tur")
    return text
'''

def strip_consecutive_newlines(text: str) -> str:
    """Strips consecutive newlines from a str with whitespace in between"""
    return re.sub(r"\s*\n\s*", "\n", text)

def optimizeDf(old_df: pd.DataFrame) -> pd.DataFrame:
    df = old_df[["left", "top", "width", "text"]]
    df['left+width'] = df['left'] + df['width']
    df = df.sort_values(by=['top'], ascending=True)
    df = df.groupby(['top', 'left+width'], sort=False)['text'].sum().unstack('left+width')
    df = df.reindex(sorted(df.columns), axis=1).dropna(how='all').dropna(axis='columns', how='all')
    df = df.fillna('')
    return df

def mergeDfColumns(old_df: pd.DataFrame, threshold: int = 10, rotations: int = 5) -> pd.DataFrame:
  df = old_df.copy()
  for j in range(0, rotations):
    new_columns = {}
    old_columns = df.columns
    i = 0
    while i < len(old_columns):
      if i < len(old_columns) - 1:
        # If the difference between consecutive column names is less than the threshold
        if any(old_columns[i+1] == old_columns[i] + x for x in range(1, threshold)):
          new_col = df[old_columns[i]].astype(str) + df[old_columns[i+1]].astype(str)
          new_columns[old_columns[i+1]] = new_col
          i = i + 1
        else: # If the difference between consecutive column names is greater than or equal to the threshold
          new_columns[old_columns[i]] = df[old_columns[i]]
      else: # If the current column is the last column
        new_columns[old_columns[i]] = df[old_columns[i]]
      i += 1
    df = pd.DataFrame.from_dict(new_columns).replace('', np.nan).dropna(axis='columns', how='all')
  return df.replace(np.nan, '')

def mergeDfRows(old_df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    new_df = old_df.iloc[:1]
    for i in range(1, len(old_df)):
        # If the difference between consecutive index values is less than the threshold
        if abs(old_df.index[i] - old_df.index[i - 1]) < threshold: 
            new_df.iloc[-1] = new_df.iloc[-1].astype(str) + old_df.iloc[i].astype(str)
        else: # If the difference is greater than the threshold, append the current row
            new_df = new_df.append(old_df.iloc[i])
    return new_df.reset_index(drop=True)

def clean_df(df):
    # Remove columns with all cells holding the same value and its length is 0 or 1
    df = df.loc[:, (df != df.iloc[0]).any()]
    # Remove rows with empty cells or cells with only the '|' symbol
    df = df[(df != '|') & (df != '') & (pd.notnull(df))]
    # Remove columns with only empty cells
    df = df.dropna(axis=1, how='all')
    return df.fillna('')


def img2txt(bytesio_image: BytesIO):
    """
    Reads image from BytesIO object. Converts it to gray scale.
    Apply Gaussian blur, followed by Otsu's threshold to obtain a binary image.
    Apply morphological operations to remove noise and artifacts.

    Supported file types: PNG, JPEG, TIFF, JPEG 2000, GIF, WebP, BMP, PNM

    How to Build `tesseract`:
        brew install wget
        brew install tesseract
        brew install tesseract-lang
        wget -O /usr/local/codebase/documentSummarizer/tur.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/tur.traineddata

    Parameters:
        bytesio_image: BytesIO object containing the image data
    Return:
        text: a str of text on the image
    """
    
    pil_image = Image.open(bytesio_image)
    image = np.array(pil_image)
    # if there is alpha channel, ignore it
    if image.shape[2] == 4:
        image = image[:, :, :3]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0,0,0), 2)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (0,0,0), 3)

    # Dilate to connect text and remove dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:
            cv2.drawContours(dilate, [c], -1, (0,0,0), -1)

    # Bitwise-and to reconstruct image
    result = cv2.bitwise_and(image, image, mask=dilate)
    result[dilate==0] = (255,255,255)

    data = pytesseract.image_to_string(result, lang="tur")

    return data

    
