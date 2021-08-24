from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

context = input("\nInput sample passage: ")

from transformers import AutoTokenizer


while True:
	questions = [
	    input("\nEnter Question - ")    
	]

	print()

	tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
	tokenizer.encode(questions[0], truncation = True, padding = True)

	from transformers import pipeline
	nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
	output = nlp({
	    'question': questions[0],
	    'context': context
	})
	print("The answer is - ", output["answer"], "\n")