sinhalese_chars = [
    "අ", "ආ", "ඇ", "ඈ", "ඉ", "ඊ",
    "උ", "ඌ", "ඍ", "ඎ", "ඏ", "ඐ",
    "එ", "ඒ", "ඓ", "ඔ", "ඕ", "ඖ",
    "ං", "ඃ",
    "ක", "ඛ", "ග", "ඝ", "ඞ", "ඟ",
    "ච", "ඡ", "ජ", "ඣ", "ඤ", "ඥ", "ඦ",
    "ට", "ඨ", "ඩ", "ඪ", "ණ", "ඬ",
    "ත", "ථ", "ද", "ධ", "න", "ඳ",
    "ප", "ඵ", "බ", "භ", "ම", "ඹ",
    "ය", "ර", "ල", "ව",
    "ශ", "ෂ", "ස", "හ", "ළ", "ෆ",
    "෴", "\u200d"
]
# "\u200d" is used with "යංශය" - කාව්‍ය, "රේඵය" - වර්‍තමාන, "Both" - මහාචාර්‍ය්‍ය, "රකාරාංශය" - මුද්‍රණය

sinhalese_vowel_signs = ["්", "ා", "ැ", "ෑ", "ි", "ී", "ු", "ූ", "ෘ", "ෙ", "ේ", "ෛ", "ො", "ෝ",
                         "ෞ", "ෟ", "ෲ", "ෳ", "ර්‍"]

vowel_sign_fix_dict = {
    "ෑ": "ැ",
    "ෙ" + "්": "ේ",
    "්" + "ෙ": "ේ",

    "ෙ" + "ා": "ො",
    "ා" + "ෙ": "ො",

    "ේ" + "ා": "ෝ",
    "ො" + "්": "ෝ",

    "ෙෙ": "ෛ",
    "ෘෘ": "ෲ",

    "ෙ" + "ෟ": "ෞ",
    "ෟ" + "ෙ": "ෞ",

    "ි" + "ී": "ී",
    "ී" + "ි": "ී",

    # duplicating same symbol
    "ේ" + "්": "ේ",
    "ේ" + "ෙ": "ේ",

    "ො" + "ා": "ො",
    "ො" + "ෙ": "ො",

    "ෝ" + "ා": "ෝ",
    "ෝ" + "්": "ෝ",
    "ෝ" + "ෙ": "ෝ",
    "ෝ" + "ේ": "ෝ",
    "ෝ" + "ො": "ෝ",

    "ෞ" + "ෟ": "ෞ",
    "ෞ" + "ෙ": "ෞ",

    # special cases - may be typing mistakes
    "ො" + "ෟ": "ෞ",
    "ෟ" + "ො": "ෞ",
}


def is_sinhalese_letter(char: str) -> bool:
    return char in sinhalese_chars


def is_sinhalese_vowel(char: str) -> bool:
    return char in sinhalese_vowel_signs


def get_fixed_vowel(vowel: str) -> str:
    return vowel_sign_fix_dict[vowel]


class SinhaleseVowelLetterFixer:
    """
    Sinhalese Language Vowel Letter Fixer
    """

    @staticmethod
    def get_fixed_text(text: str) -> str:
        """
        Fix wrong usage of vowels
        :param text: text to be fixed
        :return: fixed text with proper vowels
        """
        fixed_text = ""
        last_letter = ""
        last_vowel = ""

        for letter in text:
            if is_sinhalese_letter(letter):
                fixed_text += (last_letter + last_vowel)
                last_letter = letter
                last_vowel = ""
            elif is_sinhalese_vowel(letter):
                if last_letter == "":
                    print("Error : First letter can't be a vowel sign : " + letter)
                if last_vowel == "":
                    last_vowel = letter
                else:
                    try:
                        last_vowel = get_fixed_vowel(last_vowel + letter)
                        print(last_vowel)
                    except KeyError:
                        # fix error of mistakenly duplicate vowel
                        if last_vowel == letter:
                            continue
                        else:
                            print("Error : can't fix vowel combination " + last_vowel + " + " + letter)
            else:
                fixed_text += (last_letter + last_vowel + letter)
                last_letter = ""
                last_vowel = ""

        fixed_text += last_letter + last_vowel
        return fixed_text
