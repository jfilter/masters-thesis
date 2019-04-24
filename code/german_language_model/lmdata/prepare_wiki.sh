
dlpath="/mnt/data/group07/johannes/germanlm/dewiki.xml.bz2"
outpath="/mnt/data/group07/johannes/germanlm/proc/wiki/"

if [[ "$@" == "0" ]]
then
	wget https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2 -O $dlpath
fi
if [[ "$@" == "1" ]]
then 
	python wikiextractor/WikiExtractor.py -s --json -o $outpath --processes 8 --min_text_length 500 --filter_disambig_pages $dlpath 
fi
