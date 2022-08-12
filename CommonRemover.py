from library.BiLstm.Utils import *

csPhraseFile = './data/csTerms.csv'
nonCsPhraseFile = './data/nonCsTerms.csv'

csPhraseList = set(getDataListFromFile(csPhraseFile))
nonCsPhraseList = set(getDataListFromFile(nonCsPhraseFile))

for phrase in getCommonPhrases(csPhraseList, nonCsPhraseList):
	print("\n")
	print("Phrase: ", phrase)
	choice = input("Put 0 for non-CS and 1 for CS: ")
	while choice not in ['0','1']:
		choice = input("Put 0 for non-CS and 1 for CS: ")
	if choice == '0':
		csPhraseList.discard(phrase)
	else:	#'1'
		nonCsPhraseList.discard(phrase)

csPhraseList = list(csPhraseList)
nonCsPhraseList = list(nonCsPhraseFile)

csPhraseList.sort(key=lambda x : x.lower())
nonCsPhraseList.sort(key=lambda x : x.lower())

pd.DataFrame({
	'phrase': csPhraseList,
	}).to_csv(csPhraseFile, index=False, header=True)
pd.DataFrame({
	'phrase': nonCsPhraseList,
	}).to_csv(nonCsPhraseFile, index=False, header=True)