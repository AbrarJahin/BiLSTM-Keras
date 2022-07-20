import wikipedia
import csv
import io
import json
import pandas as pd
import nltk
import random
wikipedia.set_lang("en")

def getArticleList(phraseList):
	output, errorList = {}, []
	for phrase in phraseList:
		# Error found in "dimensional theories"
		try:
			wikiSearch = wikipedia.search(phrase, results = 100)
			wikiEntryNameFromSearch = wikiSearch[0]
			wikiPage = wikipedia.page(wikiEntryNameFromSearch)
			output[phrase] = {}
			output[phrase]['found'] = wikiPage
		except:
			print("Error", phrase)
			errorList.append(phrase)
			output[phrase] = {}
	return output, errorList