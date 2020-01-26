import sentencepiece as spm
import sys

sp = spm.SentencePieceProcessor()
sp.Load(sys.argv[4])
input_name, output_name = sys.argv[1], sys.argv[2]
file_num = int(sys.argv[3]) 
with open(input_name, "r") as f1:
    lines = f1.readlines()
    linecount = len(lines)
    batch_size = len(lines) // file_num 
    for i in range(file_num):
        f2 = open(output_name[:-4] + '_' + str(i) + '.txt', "w+")
        for j in range(batch_size):
            line = lines[i * batch_size + j]
            f2.write(" ".join(sp.EncodeAsPieces(line)))

    for i in range(file_num * batch_size, linecount):
        line = lines[i]
        f2.write(" ".join(sp.EncodeAsPieces(line)))
print("encoded")

