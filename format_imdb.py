import os
# train_positive = 
# train_negative = 


train_positive_text = []
train_negative_text = []

for curr_file in os.listdir("imdb/train/pos/"):
	with open("imdb/train/pos/"+curr_file,encoding='utf-8', errors='ignore') as f:
		lines = [line.rstrip() for line in f]
	train_positive_text.extend(lines)

for curr_file in os.listdir("imdb/train/neg/"):
	with open("imdb/train/neg/"+curr_file, encoding='utf-8',errors='ignore') as f:
		lines=[line.rstrip() for line in f]
	train_negative_text.extend(lines)


test_text = []

for curr_file in os.listdir("imdb/test/pos/"):
	with open("imdb/test/pos/"+curr_file,encoding='utf-8', errors='ignore') as f:
		lines = [line.rstrip() for line in f]
	test_text.extend(lines)

for curr_file in os.listdir("imdb/test/neg/"):
	with open("imdb/test/neg/"+curr_file, encoding="utf-8", errors="ignore") as f:
		lines = [line.rstrip() for line in f]
	test_text.extend(lines)

threshold = int(0.85*len(train_positive_text))
dev_positive_text = train_positive_text[threshold:]
dev_negative_text = train_negative_text[threshold:]

train_positive_text = train_positive_text[:threshold]
train_negative_text = train_negative_text[:threshold]



with open('imdb_train.tsv', 'w') as f:
	to_write = ""
	for i in range(len(train_positive_text)):
		to_write+=train_positive_text[i].replace("\n", " ").replace("\t"," ")+"\t"+"1"+"\n"
	for i in range(len(train_negative_text)):
		to_write+=train_negative_text[i].replace("\n", " ").replace("\t"," ")+"\t"+"0"+"\n"
	f.write(to_write)

with open('imdb_dev.tsv', 'w') as f:
	to_write = ""
	for i in range(len(dev_positive_text)):
		to_write+=dev_positive_text[i].replace("\n", " ").replace("\t"," ")+"\t"+"1"+"\n"
	for i in range(len(dev_negative_text)):
		to_write+=dev_negative_text[i].replace("\n", " ").replace("\t"," ")+"\t"+"0"+"\n"
	f.write(to_write)

with open('imdb_test.tsv', 'w') as f:
	to_write =""
	for i in range(len(test_text)):
		to_write+=test_text[i].replace("\n", " ").replace("\t"," ")+"\n"
	f.write(to_write)