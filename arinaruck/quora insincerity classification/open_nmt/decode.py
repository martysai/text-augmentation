import sentencepiece as spm
import sys

sp = spm.SentencePieceProcessor()
sp.Load("/Users/arinaruck/Downloads/transformer-ende-wmt-pyOnmt/sentencepiece.model")
input_name, output_name = sys.argv[1], sys.argv[2]
file_num = int(sys.argv[3])
with open(output_name, "w+") as f1:
    for i in range(file_num):
         f2 = open(input_name[:-4] + '_' + str(i) + '.txt', "r")
         for line in f2.readlines():         
             f1.write(sp.DecodePieces(line.split(' ')))
print("decoded")

