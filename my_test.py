from rouge import FilesRouge

hyp_path="notebooks/gen_text_zm/decoded.txt"
ref_path="notebooks/gen_text_zm/reference.txt"
files_rouge = FilesRouge()
scores = files_rouge.get_scores(hyp_path, ref_path)
# print(scores)
# or
scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
print(scores)