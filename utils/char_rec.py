from pytesseract import image_to_string,image_to_data,image_to_boxes,image_to_osd


def char_rec(crop):
    _config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789'
    boxes=image_to_boxes(crop,lang='eng',config=_config)
    lpText=image_to_string(crop,lang='eng',config=_config)
    print('a')
    print(lpText[0:len(lpText)-1])
    print('a')
    # while not lpText[0].isdigit():
    #     lpText=lpText[1:len(lpText)]
    # while not (lpText[len(lpText)-1].isdigit()):
    #     lpText=lpText[0:len(lpText)-1]
    # if lpText[0:2].isdigit():
    #     lpText=lpText[1:len(lpText)]
    return lpText,boxes