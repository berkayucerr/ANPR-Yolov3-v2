from pytesseract import image_to_string

def char_rec(crop):
    lpText=image_to_string(crop,lang='eng',config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789')
    if len(lpText)>6:
        while not lpText[0].isdigit():
            lpText=lpText[1:len(lpText)]
        while not (lpText[len(lpText)-1].isdigit()):
            lpText=lpText[0:len(lpText)-1]
        if lpText[0:2].isdigit():
            lpText=lpText[1:len(lpText)]
    return lpText