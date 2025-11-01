from datasets import load_dataset

ds_eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
ds_lao = load_dataset("openlanguagedata/flores_plus", "lao_Laoo")
ds_khm = load_dataset("openlanguagedata/flores_plus", "khm_Khmr")
ds_tha = load_dataset("openlanguagedata/flores_plus", "tha_Thai")


val_dir = "val_data"
test_dir = "test_data"

def save_to_file(dataset, lang_code, split, directory):
    file_name = f"{directory}/{split}_{lang_code}.txt"

    with open(file_name, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item["text"] + "\n")

    
# Save validation data
save_to_file(ds_eng["dev"], "english", "dev", val_dir)
save_to_file(ds_lao["dev"], "lao", "dev", val_dir)
save_to_file(ds_khm["dev"], "khmer", "dev", val_dir)
save_to_file(ds_tha["dev"], "thai", "dev", val_dir)

# # Save test data
save_to_file(ds_eng["devtest"], "english", "devtest", test_dir)
save_to_file(ds_lao["devtest"], "lao", "devtest", test_dir)
save_to_file(ds_khm["devtest"], "khmer", "devtest", test_dir)
save_to_file(ds_tha["devtest"], "thai", "devtest", test_dir)