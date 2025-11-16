#!/usr/bin/env python3
import re


DEF_REPLACEMENTS = {
    "A": "aye",
    "B": "bee",
    "C": "see",
    "D": "dee",
    "E": "ee",
    "F": "ef",
    "G": "gee",
    "H": "eich",
    "I": "eye",
    "J": "djei",
    "K": "kei",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "oh",
    "P": "pee",
    "Q": "que",
    "R": "ar",
    "S": "es",
    "T": "ti",
    "U": "you",
    "V": "vee",
    "W": "double-vee",
    "X": "ex",
    "Y": "why",
    "Z": "ze"
}


def replace_capital_letters(string, replacements=None, join_letter=' '):
    """
    Replace sequences of 2 or more capital letters
    in the input string using a dictionary.

    Args:
        string (str): The input string.
        replacements (dict): A dictionary where keys are single-letter strings
                             and values are replacement strings.

    Returns:
        str: The modified string with replacements applied.
    """

    if replacements is None:
        replacements = DEF_REPLACEMENTS
    # Find all sequences of 2 or more capital letters using regular expression
    matches = re.findall(r'\b[A-Z]{2,}\b', string)

    # Replace each match in the string with the value from the
    # dictionary one-by-one
    modified_string = ''
    last_end = 0

    for match in matches:
        start, end = re.search(match, string[last_end:]).span()
        start += last_end
        end += last_end

        # Add the part of the string before the match
        modified_string += string[last_end:start]

        # Replace each letter in the sequence with the
        # replacement letter and join
        replacements_list = []
        for char in match:
            if char in replacements:
                replacement = replacements[char]
                if isinstance(replacement, str):
                    replacements_list.append(replacement)
                else:
                    raise ValueError(
                        f"Replacement for '{char}' is not"
                        f"a string: {replacement}")
            else:
                # If no replacement is available, keep the original letter
                replacements_list.append(char)

        modified_string += join_letter.join(replacements_list)

        last_end = end

    # Add the part of the string after all matches
    modified_string += string[last_end:]

    return modified_string


# Example usage
if __name__ == "__main__":
    string = "HelloWorldThisIsAStringWithSome CAPITAL Letters"
    replacements = {"H": "G", "L": "MM", "W": "EE"}
    print(replace_capital_letters(string, replacements))
