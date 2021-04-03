#!/usr/bin/env python.

import re

def has_log(text):
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:|\
        [Tt]raceback.{0,10}:|[Bb]acktrace.{0,10}:|[Ll]ogs?:|\d{2}:\d{2}:\d{2}|\
        (INFO|[Ii]nfo|FAIL|[Ff]ail|WARN(ING)?|[Ww]arn(ing)?|FATAL|[Ff]atal|DEBUG|[Dd]ebug|SYSTEM|[Ss]ystem|ERROR|[Ee]rror)\s*:'
    results = re.findall(pattern, text)
    return len(results) != 0

def remove_log(text):
    # If logs appear in a code block, remove the whole code block
    CODE_REGEX = r'```.+?```'
    for match in re.findall(CODE_REGEX, text, flags=re.S):
        if has_log(str(match)):
            text = text.replace(str(match), '')
    
    # If logs appear outside a code block, remove corresponding lines
    LOG_REGEX = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:|\
        [Tt]raceback.{0,10}:|[Bb]acktrace.{0,10}:|[Ll]ogs?:|\d{2}:\d{2}:\d{2}|\
        (INFO|[Ii]nfo|FAIL|[Ff]ail|WARN(ING)?|[Ww]arn(ing)?|FATAL|[Ff]atal|DEBUG|[Dd]ebug|SYSTEM|[Ss]ystem|ERROR|[Ee]rror)\s*:).+?\n'
    for match in re.findall(LOG_REGEX, text, flags=re.S):
        text = text.replace(str(match), '')

    return text

def has_code_block(text):
        CODE_REGEX = r'```.+?```'
        for match in re.findall(CODE_REGEX, text, flags=re.S):
            if not has_log(str(match)):
                return True

        return False

def remove_code_block(text):
        CODE_REGEX = r'```.+?```'
        for match in re.findall(CODE_REGEX, text, flags=re.S):
            if not has_log(str(match)):
                text = text.replace(str(match), '')
        return text

def has_url(text):
    pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    results = re.findall(pattern, text)
    return len(results) != 0

def remove_url(text):
    # remove link in markdown
    markdown_pattern = r'(?:__|[*#])|\[(.*?)\]\(.*?\)'
    replacement_alt_text = r'\1'
    result_1 = re.sub(markdown_pattern, replacement_alt_text, text)
    
    # remove link in plain text
    plain_pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    replacement_plain = ''
    result_2 = re.sub(plain_pattern, replacement_plain, result_1)
    return result_2


def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text
