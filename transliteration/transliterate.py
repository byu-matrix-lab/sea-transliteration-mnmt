import json
import sys
import regex as re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


class Transliterator():
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self.load_json()
        self.max_key_length = max(len(key) for key in self.data["Characters"].keys())

        if "Consonants" in self.data:
            self.consonants = "".join(self.data["Consonants"])
            self.consonant_pattern = r"[" + re.escape(self.consonants) + r"]"

    def load_json(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            sys.exit(1)

    def check_substring_match(self, substring: str, rules: dict, transliteration_method: str = "ipa"):
        for key, value in rules.items():
            if "◌" in key:
                place_holder_pattern = key.replace("◌", "[{}]".format(self.consonants))
                if re.fullmatch(place_holder_pattern, substring):
                    consonant = re.search(self.consonant_pattern, substring).group(0)
                    return rules[consonant], value
        return None, None

    def transliterate(self, text: str, transliteration_method="ipa"):
        rules = self.data["Characters"]
        i = 0
        result = []
        while i < len(text):
            for length in range(self.max_key_length, 0, -1):
                if i + length > len(text):
                    continue
                substring = text[i:i + length]
                if substring in rules:
                    result.append(rules[substring][transliteration_method])
                    i += length
                    break
                consonant_values, vowel_values = self.check_substring_match(substring, rules)
                if consonant_values and vowel_values:
                    result.append(consonant_values[transliteration_method])
                    result.append(vowel_values[transliteration_method])
                    i += len(substring)
                    break
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)


def transliterate_line(line: str, json_path: str, method: str) -> str:
    """
    This function is run in a subprocess.
    A new Transliterator instance is created in each process.
    """
    transliterator = Transliterator(json_path)
    return transliterator.transliterate(line, transliteration_method=method)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python transliterate.py <json_path> <method> <text_file>")
        print("Example methods: ipa, romanized, cat")
        sys.exit(1)

    json_path = sys.argv[1]
    method = sys.argv[2]
    text_file = sys.argv[3]

    with open(text_file, 'r', encoding='utf-8') as f:
        input_lines = [line.strip() for line in f.readlines() if line.strip()]

    transliterated = [None] * len(input_lines)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(transliterate_line, line, json_path, method): i for i, line in enumerate(input_lines)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Transliterating"):
            idx = futures[future]
            transliterated[idx] = future.result()

    output_path = f"{text_file.rsplit('.', 1)[0]}_{method}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in transliterated:
            f.write(line + '\n')

    print(f"Transliterated text saved to {output_path}")
