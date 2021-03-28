#!/usr/bin/env python.

import re

# Returns True of `text` contains logs; False otherwise
def has_log(text):
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:|\
        [Tt]raceback.*:|[Bb]acktrace.*:|[Ll]ogs?:|\d{2}:\d{2}:\d{2}|\
        (INFO|info|FAIL|fail|WARN(ING)?|warn(ing)?|FATAL|fatal|DEBUG|debug|SYSTEM|system|ERROR|error)\s*:'
    results = re.findall(pattern, text)
    return len(results) != 0

def remove_log(text):
	# TODO
	pass

def has_code_block(text):
	# TODO
	pass

def remove_code_block(text):
	# TODO
	pass

def has_url(text):
    pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    results = re.findall(pattern, text)
    return len(results) != 0

def remove_url(text):
    pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    replacement = ''
    result = re.sub(pattern, replacement, text)
    return result
