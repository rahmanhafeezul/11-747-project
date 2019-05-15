import csv
import json
import pdb

train_file_name = "yelp/yelp_reviews_train.json"
test_file_name = "yelp/yelp_reviews_test.json"

def return_text_and_labels(file, mode):
	text_reviews = []
	ratings = []
	# pdb.set_trace()
	with open(file,encoding='utf-8', errors='ignore') as f:
		dev_json_lines = [line.rstrip() for line in f]

	for line in dev_json_lines:
		if(mode=="train"):
			text_reviews.append(json.loads(line)["text"])
			ratings.append(json.loads(line)['stars'])
		if(mode=="test"):
			text_reviews.append(json.loads(line)["text"])
		
	return text_reviews, ratings
train_text, train_rating = return_text_and_labels(train_file_name, "train")
threshold = int(0.85*float(len(train_text)))
dev_text = train_text[threshold:]
dev_rating = train_rating[threshold:]
train_text=train_text[:threshold]
train_rating=train_rating[:threshold]
test_text, test_rating = return_text_and_labels(test_file_name, "test")
# pdb.set_trace()

count=0
with open('yelp_train.tsv', 'w') as f:
	to_write = ""
	for i in range(len(train_text)):
		count+=1
		if(count%100==0):
			to_write+=train_text[i].replace("\n", " ")+"\t"+str(int(train_rating[i])-1)+"\n"
	f.write(to_write)

count=0
with open('yelp_dev.tsv', 'w') as f:
	to_write = ""
	for i in range(len(dev_text)):
		count+=1
		if(count%100==0):
			to_write+=dev_text[i].replace("\n", " ")+"\t"+str(int(dev_rating[i])-1)+"\n"
	f.write(to_write)

count=0
with open('yelp_test.tsv', 'w') as f:
	to_write =""
	for i in range(len(test_text)):
		count+=1
		if(count%100==0):
			to_write+=test_text[i].replace("\n", " ")+"\n"
	f.write(to_write)