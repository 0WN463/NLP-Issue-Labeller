#!/usr/bin/env python.

import re

# Returns True of `text` contains logs; False otherwise
def has_log(text):
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}:|\
        [Tt]raceback.*:|[Bb]acktrace.*:|[Ll]ogs?:|\d{2}:\d{2}:\d{2}|\
        (INFO|info|FAIL|fail|WARN(ING)?|warn(ing)?|FATAL|fatal|DEBUG|debug|SYSTEM|system)\s*:'
    results = re.findall(pattern, text)
    return len(results) != 0
