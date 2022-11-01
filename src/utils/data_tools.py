import re


def cut_sentence(text, whole=False):
    """
    :param text:
    :param whole: 是否取整句
    :return:
    """
    if whole:
        pattern = r'\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|。|、|；|‘|’|【|】|·|！| |…|（|）|？'
    else:
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）|？'
    result_list = re.split(pattern, text)
    return result_list


def just_chinese(strings):
    regStr = ".*?([\u4E00-\u9FA5]+).*?"
    expr = ''.join(re.findall(regStr, strings))
    if expr:
        return expr
    return '\n'


def clean_text(strings):
    regStr = '[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa50-9]'
    expr = re.findall(regStr, strings)
    if expr:
        return ''.join(expr)
    return '\n'
