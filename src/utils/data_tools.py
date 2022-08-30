import re


def cut_sentence(text, whole=False):
    '''

    :param text:
    :param whole: 是否取整句
    :return:
    '''
    if whole:
        pattern = r'\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|。|、|；|‘|’|【|】|·|！| |…|（|）'
    else:
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list = re.split(pattern, text)
    return result_list


def just_chinese(strings):
    regStr = ".*?([\u4E00-\u9FA5]+).*?"
    expr = ''.join(re.findall(regStr, strings))
    if expr:
        return expr
    return '\n'
