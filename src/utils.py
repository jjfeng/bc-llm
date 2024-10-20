import ast
import regex
import re

import numpy as np


def _process_to_valid_json(s):
    """
    Processes a string to make it valid JSON by fixing unclosed quotes/braces
    and removing extra characters at the end.

    Args:
        s (str): The input string that may be invalid JSON.

    Returns:
        str: A string that is valid JSON.
    """
    output_chars = []
    nesting_stack = []
    in_string = False
    escape = False

    valid_chars = set('0123456789+-eE.')
    whitespace = set(' \t\n\r')
    i = 0
    while i < len(s):
        c = s[i]
        output_chars.append(c)

        if escape:
            escape = False
            i += 1
            continue

        if c == '\\' and in_string:
            escape = True
            i += 1
            continue

        if c == '"':
            in_string = not in_string
            i += 1
            continue

        if not in_string:
            if c in whitespace or c in ',:':
                i += 1
                continue
            elif c in '{[':
                nesting_stack.append(c)
            elif c in '}]':
                if not nesting_stack:
                    # Unmatched closing brace/bracket
                    output_chars.pop()  # Remove the invalid character
                    break
                top = nesting_stack.pop()
                if c == '}' and top != '{':
                    # Mismatched braces
                    output_chars.pop()
                    break
                if c == ']' and top != '[':
                    # Mismatched brackets
                    output_chars.pop()
                    break
            elif c in valid_chars:
                # Part of a number value
                pass
            elif c in 'tfn':  # Start of true, false, null
                pass
            else:
                # Invalid character outside string
                output_chars.pop()
                break
        i += 1

    # Close any unclosed strings
    if in_string:
        output_chars.append('"')

    # Close any unclosed braces/brackets
    while nesting_stack:
        top = nesting_stack.pop()
        if top == '{':
            output_chars.append('}')
        elif top == '[':
            output_chars.append(']')

    return ''.join(output_chars)

def convert_to_list_of_jsons(text: str, logging=None) -> list:
    # Regular expression to match comments
    text = re.sub(r'//.*', '', text)  # Remove anything after "//" till the end of the line
    
    json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    json_strings = json_pattern.findall(text)
    ans = []
    for json_str in json_strings:
        ans.append(convert_to_json(json_str, logging))
    return ans



def convert_to_json(text: str, logging=None) -> dict:
    # Regular expression to match comments
    text = re.sub(r'//.*', '', text)  # Remove anything after "//" till the end of the line
    
    json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    json_strings = json_pattern.findall(text)
    json_text = {}
    for json_str in json_strings:
        try:
            json_text = json_text | ast.literal_eval(json_str)
        except Exception as e:
            try:
                json_text = json_text | ast.literal_eval(_process_to_valid_json(json_str))
            except Exception as e:
                print(e)
                print(text)
                if logging is not None:
                    logging.info("failed to parse into json %s", text)
                    logging.info(e)
    # assert len(json_text) > 0
    if logging is not None and len(json_text) == 0:
        logging.info("WARNING: no json found %s", text)
    return json_text


def get_calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform"):
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred
