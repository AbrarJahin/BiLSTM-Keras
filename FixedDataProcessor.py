import pandas as pd
#from library.BiLstm.Utils import *

# Save location - After all processing
actualCsPhraseCsv = './data/csTerms.csv'
actualNonCsPhraseCsv = './data/nonCsTerms.csv'

#This 2 files are manually entered
fixedCsPhraseCsv = './output/CsWrong.csv'
fixedNonCsPhraseCsv = './output/NonCsWrong_modified.csv'

csPhraseList = []
nonCsPhraseList = []

#Take nonCS CSV
nonCsCSV = pd.read_csv(fixedNonCsPhraseCsv, encoding= 'unicode_escape', index_col=None)

if 'IfCS' in nonCsCSV.columns:
	for index, row in nonCsCSV.iterrows():
		if row['IfCS']==1:
			csPhraseList.append(row['Phrase'])
		else:
			nonCsPhraseList.append(row['Phrase'])
else:
	#Take the whole data as array
	nonCsPhraseList.extend([x for x in nonCsCSV['Phrase'] if x!=''])

if 'Correct' in nonCsCSV.columns:
	csPhraseList.extend([x for x in nonCsCSV['Correct'] if x!=''])


#Take CS CSV
csCSV = pd.read_csv(fixedCsPhraseCsv, encoding= 'unicode_escape', index_col=None)

if 'IfCS' in csCSV.columns:
	for index, row in nonCsCSV.iterrows():
		if row['IfCS']==1:
			csPhraseList.append(row['Phrase'])
		else:
			nonCsPhraseList.append(row['Phrase'])
else:
	#Take the whole data as array
	csPhraseList.extend([x for x in nonCsCSV['Phrase'] if x!=''])

if 'Correct' in nonCsCSV.columns:
	csPhraseList.extend([x for x in nonCsCSV['Correct'] if x!=''])

newNonCsSet = set([x for x in set(nonCsPhraseList) if x!='' and isinstance(x, str)])
newCsSet = set([x for x in set(csPhraseList) if x!='' and isinstance(x, str)])

oldNonCsCSV = pd.read_csv(actualNonCsPhraseCsv, encoding= 'unicode_escape', index_col=None)
oldCsCSV = pd.read_csv(actualCsPhraseCsv, encoding= 'unicode_escape', index_col=None)

filteredCs = [x for x in oldCsCSV['phrase'] if x not in newNonCsSet and x!='' and isinstance(x, str)]
filteredNonCs = [x for x in oldNonCsCSV['phrase'] if x not in newCsSet and x!='' and isinstance(x, str)]

#Save Files
pd.DataFrame({
		'phrase': sorted(set(filteredNonCs + list(newNonCsSet)))
		}).to_csv(actualNonCsPhraseCsv, index=False, header=True)
pd.DataFrame({
		'phrase': sorted(set(filteredCs + list(newCsSet)))
		}).to_csv(actualCsPhraseCsv, index=False, header=True)